//! Live transcript: a rough, English-only preview of a recording in progress.
//!
//! The capture sidecar mirrors every frame it writes to a WAV into a headerless
//! 16 kHz mono int16 file (`live-mic.pcm`, `live-system.pcm`). A loop here
//! tails those files and, every few seconds, runs a small whisper model over
//! the recent audio of each: what has arrived since the last committed text,
//! never more than 20 seconds. Text that has settled is *committed*: sent once
//! and never changed. The tail that may still change is *provisional*:
//! replaced wholesale on every tick.
//!
//! The preview is disposable. It is never saved and never feeds the notes, and
//! nothing here may harm a recording. The loop owns its thread and its whisper
//! context, takes none of the host's locks, and turns every failure into a
//! skipped tick or an "unavailable" status.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Serialize;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::transcribe::{is_noise_annotation, strip_dialogue_dash};

/// The whisper.cpp model behind the preview (43 MB, English only), whatever
/// the user picked for the real transcript.
pub const MODEL: &str = "tiny.en-q8_0";

/// The raw PCM files the capture sidecar writes in the meeting folder: 16 kHz,
/// mono, int16, little-endian, no header. Passing `--live-pcm` turns them on.
pub const MIC_PCM: &str = "live-mic.pcm";
pub const SYSTEM_PCM: &str = "live-system.pcm";

const SAMPLES_PER_MS: i64 = 16;
/// How often each stream is transcribed.
pub const TICK: Duration = Duration::from_secs(3);
/// The most recent audio kept for a stream, and so the most whisper sees at once.
pub const WINDOW_MS: i64 = 20_000;
/// Text ending this far before the end of the audio counts as settled, so
/// whisper heard at least this much after anything it commits. The window is
/// longer than this so there is always room for text that has not settled yet.
pub const COMMIT_HORIZON_MS: i64 = 6_000;
const WINDOW_SAMPLES: usize = (WINDOW_MS * SAMPLES_PER_MS) as usize;
/// Whisper has nothing to say about less than a second of audio.
const MIN_DECODE_SAMPLES: usize = 16_000;
/// Audio quieter than this (RMS, full scale 1.0, about -50 dBFS) is treated as
/// silence: above the hiss of a quiet room, below quiet speech.
const SILENCE_RMS: f32 = 0.003;
/// Consecutive failed ticks after which the preview gives up.
const MAX_FAILURES: u32 = 3;

/// Which side of the call a line came from. The mic is "me", the system audio
/// is everyone else; no diarization runs on the live path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Stream {
    Me,
    Them,
}

/// A stretch of text and where it sits, in ms from the start of the recording.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Line {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

/// The `live-transcript` event.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LiveEvent {
    pub stream: Stream,
    /// New since the last event: append, never rewrite.
    pub committed: Vec<Line>,
    /// The moving tail: replace whatever was shown for this stream.
    pub provisional: Vec<Line>,
}

/// The `live-status` event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Fetching or loading the preview model.
    #[default]
    Preparing,
    Live,
    /// The preview gave up. The recording is unaffected.
    Unavailable,
    /// The recording ended and the last words were flushed.
    Finished,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Update {
    Status(Status),
    Transcript(LiveEvent),
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TaggedLine {
    pub stream: Stream,
    #[serde(flatten)]
    pub line: Line,
}

/// Everything the preview has said so far, for a panel that opened after some
/// of it was emitted. Small: text only, for one recording.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    pub status: Status,
    pub committed: Vec<TaggedLine>,
    pub provisional: Vec<TaggedLine>,
}

impl Snapshot {
    pub fn apply(&mut self, update: &Update) {
        match update {
            Update::Status(Status::Preparing) => *self = Snapshot::default(),
            Update::Status(status) => self.status = *status,
            Update::Transcript(event) => {
                let tag = |line: &Line| TaggedLine {
                    stream: event.stream,
                    line: line.clone(),
                };
                self.committed.extend(event.committed.iter().map(tag));
                self.provisional.retain(|l| l.stream != event.stream);
                self.provisional.extend(event.provisional.iter().map(tag));
            }
        }
    }
}

/// The outcome of splitting one tick's lines at the commit horizon.
#[derive(Debug, PartialEq, Eq)]
pub struct Split {
    /// Settled for the first time on this tick.
    pub committed: Vec<Line>,
    /// Still free to change.
    pub provisional: Vec<Line>,
    /// The new watermark. Never lower than the one passed in.
    pub committed_ms: i64,
}

/// Split whisper's lines for a window covering `window_start_ms..window_end_ms`
/// (absolute times, ms from the start of the recording) into text that has
/// settled and text that has not.
///
/// Lines ending at least `COMMIT_HORIZON_MS` before the window's end commit,
/// provided everything before them has. A line straddling the horizon stays
/// provisional whole: its words cannot be cut at the horizon, and it will have
/// moved behind it a tick or two later.
///
/// Whisper cuts the same audio differently from tick to tick, so a line whose
/// middle lies at or before the watermark is a repeat of something already
/// committed and is dropped. Times are clamped to the window, since whisper
/// will sometimes put a segment past the end of the audio it was given.
pub fn split_commit(
    lines: &[Line],
    window_start_ms: i64,
    window_end_ms: i64,
    committed_ms: i64,
) -> Split {
    split_at(
        lines,
        window_start_ms,
        window_end_ms,
        window_end_ms - COMMIT_HORIZON_MS,
        committed_ms,
    )
}

/// Like `split_commit`, for the end of the recording: nothing is left to
/// change, so everything not yet committed commits.
pub fn flush_commit(
    lines: &[Line],
    window_start_ms: i64,
    window_end_ms: i64,
    committed_ms: i64,
) -> Split {
    split_at(
        lines,
        window_start_ms,
        window_end_ms,
        window_end_ms,
        committed_ms,
    )
}

fn split_at(
    lines: &[Line],
    window_start_ms: i64,
    window_end_ms: i64,
    horizon_ms: i64,
    committed_ms: i64,
) -> Split {
    let mut split = Split {
        committed: Vec::new(),
        provisional: Vec::new(),
        committed_ms,
    };
    if window_end_ms <= window_start_ms {
        return split;
    }
    // Once one line stays provisional, so does everything after it: committed
    // text must never be followed by older text.
    let mut settled = true;
    for line in lines {
        let start_ms = line.start_ms.clamp(window_start_ms, window_end_ms);
        let end_ms = line.end_ms.clamp(start_ms, window_end_ms);
        if (start_ms + end_ms) / 2 <= committed_ms {
            continue;
        }
        let line = Line {
            start_ms,
            end_ms,
            text: line.text.clone(),
        };
        if settled && end_ms <= horizon_ms {
            split.committed_ms = split.committed_ms.max(end_ms);
            split.committed.push(line);
        } else {
            settled = false;
            split.provisional.push(line);
        }
    }
    split
}

/// Samples read from a PCM file, and where to carry on from.
#[derive(Debug, PartialEq)]
pub struct PcmChunk {
    /// Index, in samples from the start of the file, of `samples[0]`.
    pub first_sample: u64,
    /// -1.0..1.0.
    pub samples: Vec<f32>,
    /// Pass this as `offset` next time.
    pub next_offset: u64,
}

/// Read what has been appended to a headerless 16-bit little-endian PCM file
/// past byte `offset`, at most `max_samples` of it (the newest, if there is
/// more).
///
/// The file is still being written. Only whole samples are returned, so a
/// trailing odd byte waits for its other half. A missing or unreadable file, or
/// one shorter than `offset`, simply has nothing new.
pub fn read_pcm_since(path: &Path, offset: u64, max_samples: usize) -> PcmChunk {
    let nothing = PcmChunk {
        first_sample: offset / 2,
        samples: Vec::new(),
        next_offset: offset,
    };
    let Ok(mut file) = File::open(path) else {
        return nothing;
    };
    let Ok(len) = file.metadata().map(|m| m.len()) else {
        return nothing;
    };
    let available = len.saturating_sub(offset) & !1;
    if available == 0 {
        return nothing;
    }
    // Far behind (say the model took a while to download): skip to the newest.
    let wanted = available.min(max_samples as u64 * 2);
    let start = offset + (available - wanted);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return nothing;
    }
    let mut bytes = Vec::with_capacity(wanted as usize);
    // A short read just means fewer samples this time.
    let _ = file.take(wanted).read_to_end(&mut bytes);
    let whole = bytes.len() & !1;
    if whole == 0 {
        return nothing;
    }
    PcmChunk {
        first_sample: start / 2,
        samples: bytes[..whole]
            .chunks_exact(2)
            .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32768.0)
            .collect(),
        next_offset: start + whole as u64,
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// One stream's place in its PCM file, and what has been committed from it.
#[derive(Debug, Default)]
pub struct StreamTail {
    /// Bytes consumed from the PCM file.
    offset: u64,
    /// The newest audio, at most `WINDOW_MS` of it.
    window: Vec<f32>,
    /// Index, in samples from the start of the recording, of `window[0]`.
    window_start: u64,
    /// Nothing at or before this (ms from the start of the recording) is sent
    /// again.
    committed_ms: i64,
}

impl StreamTail {
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn committed_ms(&self) -> i64 {
        self.committed_ms
    }

    pub fn window(&self) -> &[f32] {
        &self.window
    }

    fn window_end(&self) -> u64 {
        self.window_start + self.window.len() as u64
    }

    /// The window's start and end, in ms from the start of the recording.
    pub fn window_ms(&self) -> (i64, i64) {
        (
            self.window_start as i64 / SAMPLES_PER_MS,
            self.window_end() as i64 / SAMPLES_PER_MS,
        )
    }

    /// Pull in whatever the capture has written since the last call and drop
    /// audio older than the window.
    pub fn ingest(&mut self, path: &Path) {
        let chunk = read_pcm_since(path, self.offset, WINDOW_SAMPLES);
        self.offset = chunk.next_offset;
        if chunk.samples.is_empty() {
            return;
        }
        if chunk.first_sample != self.window_end() {
            // The reader skipped ahead; what we held is no longer adjacent.
            self.window.clear();
            self.window_start = chunk.first_sample;
        }
        self.window.extend_from_slice(&chunk.samples);
        if self.window.len() > WINDOW_SAMPLES {
            let excess = self.window.len() - WINDOW_SAMPLES;
            self.window.drain(..excess);
            self.window_start += excess as u64;
        }
    }

    /// The part of the window after the watermark, with the time it starts at
    /// (ms from the start of the recording): the only audio that can still
    /// produce new text, and the only audio whisper is given.
    pub fn uncommitted(&self) -> (i64, &[f32]) {
        let (window_start_ms, _) = self.window_ms();
        let skip = ((self.committed_ms - window_start_ms).max(0) * SAMPLES_PER_MS) as usize;
        let skip = skip.min(self.window.len());
        (
            window_start_ms + skip as i64 / SAMPLES_PER_MS,
            &self.window[skip..],
        )
    }
}

/// Turns audio into lines.
pub trait Decoder {
    /// Transcribe 16 kHz mono samples. Times are ms from the start of
    /// `samples`.
    fn decode(&mut self, samples: &[f32]) -> Result<Vec<Line>, String>;
}

/// The preview model: whisper `tiny.en`, loaded once and reused for every tick
/// of both streams.
pub struct WhisperDecoder {
    state: WhisperState,
    _context: WhisperContext,
}

impl WhisperDecoder {
    pub fn load(model: &Path) -> Result<Self, String> {
        let context = WhisperContext::new_with_params(
            model.to_str().ok_or("model path is not valid utf-8")?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| format!("failed to load whisper model: {e}"))?;
        let state = context.create_state().map_err(|e| e.to_string())?;
        Ok(Self {
            state,
            _context: context,
        })
    }
}

impl Decoder for WhisperDecoder {
    fn decode(&mut self, samples: &[f32]) -> Result<Vec<Line>, String> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // Every tick decodes a fresh window, so the previous tick's text must
        // not prime this one, and a hard stretch is not worth retrying hotter.
        params.set_no_context(true);
        params.set_temperature_inc(0.0);

        self.state
            .full(params, samples)
            .map_err(|e| format!("preview transcription failed: {e}"))?;

        let mut lines = Vec::new();
        for segment in self.state.as_iter() {
            let text = segment.to_str_lossy().map_err(|e| e.to_string())?;
            let text = strip_dialogue_dash(text.trim()).to_string();
            if text.is_empty() || is_noise_annotation(&text) {
                continue;
            }
            // whisper timestamps are in centiseconds
            lines.push(Line {
                start_ms: segment.start_timestamp() * 10,
                end_ms: segment.end_timestamp() * 10,
                text,
            });
        }
        Ok(lines)
    }
}

/// Whether the audio under a line (relative to the window) is silent, which
/// means whisper made the words up.
fn is_silent_line(window: &[f32], line: &Line) -> bool {
    let from = (line.start_ms.max(0) * SAMPLES_PER_MS) as usize;
    if from >= window.len() {
        return true;
    }
    let to = ((line.end_ms.max(0) * SAMPLES_PER_MS) as usize).min(window.len());
    from < to && rms(&window[from..to]) < SILENCE_RMS
}

struct Channel {
    stream: Stream,
    path: PathBuf,
    tail: StreamTail,
    /// The provisional lines last sent.
    shown: Vec<Line>,
}

impl Channel {
    /// One pass over this stream. `flush` is the last one, which commits
    /// everything. The event, if any, is returned even when `Err` is, since
    /// a failed final pass still commits what was last shown.
    fn step<D: Decoder>(
        &mut self,
        decoder: &mut D,
        flush: bool,
    ) -> (Option<LiveEvent>, Option<String>) {
        self.tail.ingest(&self.path);
        let (_, window_end_ms) = self.tail.window_ms();
        // Whisper hears only what comes after the watermark. Handing it
        // committed audio again would have it cut that audio differently, and
        // words that belong after the watermark would then arrive inside a line
        // that looks like a repeat of committed text and be dropped with it.
        let (region_start_ms, region) = self.tail.uncommitted();

        // Nothing new to say: no speech since the watermark. Words still on
        // show came from audio that has since gone quiet or been committed.
        if rms(region) < SILENCE_RMS {
            return (self.settle(Vec::new(), Vec::new()), None);
        }
        // Too little to be worth a pass yet; more is on its way. The last pass
        // has no more coming, so it pads with silence instead.
        let padded;
        let audio = if region.len() >= MIN_DECODE_SAMPLES {
            region
        } else if flush {
            let mut samples = region.to_vec();
            samples.resize(MIN_DECODE_SAMPLES, 0.0);
            padded = samples;
            &padded[..]
        } else {
            return (None, None);
        };

        let lines = match decoder.decode(audio) {
            Ok(lines) => lines,
            Err(message) => {
                let event = if flush {
                    let shown = std::mem::take(&mut self.shown);
                    let split = flush_commit(
                        &shown,
                        region_start_ms,
                        window_end_ms,
                        self.tail.committed_ms,
                    );
                    self.tail.committed_ms = split.committed_ms;
                    self.settle(split.committed, Vec::new())
                } else {
                    None
                };
                return (event, Some(message));
            }
        };

        let lines: Vec<Line> = lines
            .into_iter()
            .filter(|line| !is_silent_line(region, line))
            .map(|line| Line {
                start_ms: line.start_ms + region_start_ms,
                end_ms: line.end_ms + region_start_ms,
                text: line.text,
            })
            .collect();
        let committed_ms = self.tail.committed_ms;
        let split = if flush {
            flush_commit(&lines, region_start_ms, window_end_ms, committed_ms)
        } else {
            split_commit(&lines, region_start_ms, window_end_ms, committed_ms)
        };
        self.tail.committed_ms = split.committed_ms;
        (self.settle(split.committed, split.provisional), None)
    }

    /// Record what is now on show and build the event, unless nothing changed.
    fn settle(&mut self, committed: Vec<Line>, provisional: Vec<Line>) -> Option<LiveEvent> {
        let changed = !committed.is_empty() || provisional != self.shown;
        self.shown = provisional.clone();
        changed.then_some(LiveEvent {
            stream: self.stream,
            committed,
            provisional,
        })
    }
}

/// What one pass over both streams produced.
#[derive(Debug, Default)]
pub struct Pass {
    pub events: Vec<LiveEvent>,
    /// The first decode error, if any. The pass still carries what it could.
    pub error: Option<String>,
}

/// The transcription state for one recording: both streams, one decoder.
pub struct Preview<D: Decoder> {
    decoder: D,
    channels: [Channel; 2],
}

impl<D: Decoder> Preview<D> {
    /// Tail `live-mic.pcm` and `live-system.pcm` in `dir`.
    pub fn new(decoder: D, dir: &Path) -> Self {
        let channel = |stream, name: &str| Channel {
            stream,
            path: dir.join(name),
            tail: StreamTail::default(),
            shown: Vec::new(),
        };
        Self {
            decoder,
            channels: [
                channel(Stream::Me, MIC_PCM),
                channel(Stream::Them, SYSTEM_PCM),
            ],
        }
    }

    /// Transcribe what has arrived since the last pass.
    pub fn tick(&mut self) -> Pass {
        self.pass(false)
    }

    /// The last pass, once the capture has stopped writing: whatever is still
    /// provisional commits.
    pub fn flush(&mut self) -> Pass {
        self.pass(true)
    }

    fn pass(&mut self, flush: bool) -> Pass {
        let mut pass = Pass::default();
        for channel in &mut self.channels {
            let (event, error) = channel.step(&mut self.decoder, flush);
            pass.events.extend(event);
            if pass.error.is_none() {
                pass.error = error;
            }
        }
        pass
    }
}

/// Delete the raw PCM files from a meeting folder.
pub fn remove_pcm_files(dir: &Path) {
    for name in [MIC_PCM, SYSTEM_PCM] {
        let _ = std::fs::remove_file(dir.join(name));
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// How the loop is told to stop, ordered by how final it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    Running,
    /// The recording ended: do one last pass, then stop.
    Finish,
    /// Stop now (panel closed, or the recording was thrown away).
    Cancel,
}

struct Control {
    phase: Mutex<Phase>,
    wake: Condvar,
    /// Still fetching or loading the model, so there is nothing to flush.
    preparing: AtomicBool,
}

impl Control {
    fn set(&self, next: Phase) {
        let mut phase = lock(&self.phase);
        if next > *phase {
            *phase = next;
        }
        self.wake.notify_all();
    }

    fn phase(&self) -> Phase {
        *lock(&self.phase)
    }

    /// Sleep until `deadline` or until told to stop, whichever is first.
    fn wait_until(&self, deadline: Instant) -> Phase {
        let mut phase = lock(&self.phase);
        loop {
            let now = Instant::now();
            if *phase != Phase::Running || now >= deadline {
                return *phase;
            }
            phase = self
                .wake
                .wait_timeout(phase, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

/// Applies each update to the snapshot, then hands it to the host to emit. The
/// snapshot lock is never held across the emit.
struct Publisher<E> {
    snapshot: Arc<Mutex<Snapshot>>,
    emit: E,
}

impl<E: Fn(&Update)> Publisher<E> {
    fn publish(&self, update: Update) {
        lock(&self.snapshot).apply(&update);
        (self.emit)(&update);
    }

    fn status(&self, status: Status) {
        self.publish(Update::Status(status));
    }

    fn events(&self, events: Vec<LiveEvent>) {
        for event in events {
            self.publish(Update::Transcript(event));
        }
    }
}

/// How the loop paces itself.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub tick: Duration,
    /// Minimum rest between passes. A pass that overruns the tick is not
    /// followed by a flat-out one, so a slow machine does less, not more.
    pub min_rest: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            tick: TICK,
            min_rest: Duration::from_secs(1),
        }
    }
}

/// A running preview. Dropping it stops the loop at once.
pub struct Handle {
    control: Arc<Control>,
    thread: Option<JoinHandle<()>>,
}

impl Handle {
    /// The recording has ended and its PCM files are complete: do a last pass
    /// that commits the provisional text, and wait for it, but no longer than
    /// `timeout`. Past that the loop is told to stop and left behind, so the
    /// caller is never held up by it.
    pub fn finish(mut self, timeout: Duration) {
        if self.control.preparing.load(Ordering::SeqCst) {
            return; // nothing was transcribed, so nothing to flush; drop cancels
        }
        self.control.set(Phase::Finish);
        if let Some(thread) = self.thread.take() {
            let deadline = Instant::now() + timeout;
            while !thread.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            if thread.is_finished() {
                let _ = thread.join();
            }
        }
    }

    /// Stop the loop without waiting for it.
    pub fn cancel(self) {}

    /// A way to stop the loop from somewhere that does not own the handle.
    pub fn canceller(&self) -> Canceller {
        Canceller(self.control.clone())
    }
}

/// Stops a running preview at once, like dropping its `Handle`, without a last
/// pass. Cheap to clone and safe to use after the handle is gone.
#[derive(Clone)]
pub struct Canceller(Arc<Control>);

impl Canceller {
    pub fn cancel(&self) {
        self.0.set(Phase::Cancel);
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.control.set(Phase::Cancel);
    }
}

/// Start the preview on its own thread.
///
/// `prepare` runs there first, so a slow model download never delays the
/// caller, and returns the decoder. `snapshot` is updated before each `emit`,
/// which receives every status change and every transcript event.
pub fn spawn<D, P, E>(prepare: P, dir: PathBuf, snapshot: Arc<Mutex<Snapshot>>, emit: E) -> Handle
where
    D: Decoder + 'static,
    P: FnOnce() -> Result<D, String> + Send + 'static,
    E: Fn(&Update) + Send + 'static,
{
    spawn_with(Timing::default(), prepare, dir, snapshot, emit)
}

/// `spawn` with explicit pacing.
pub fn spawn_with<D, P, E>(
    timing: Timing,
    prepare: P,
    dir: PathBuf,
    snapshot: Arc<Mutex<Snapshot>>,
    emit: E,
) -> Handle
where
    D: Decoder + 'static,
    P: FnOnce() -> Result<D, String> + Send + 'static,
    E: Fn(&Update) + Send + 'static,
{
    let control = Arc::new(Control {
        phase: Mutex::new(Phase::Running),
        wake: Condvar::new(),
        preparing: AtomicBool::new(true),
    });
    let out = Publisher { snapshot, emit };
    let spawned = std::thread::Builder::new()
        .name("live-transcript".into())
        .spawn({
            let control = control.clone();
            move || {
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    run(timing, prepare, &dir, &control, &out)
                }));
                if outcome.is_err() {
                    eprintln!("live transcript stopped: the preview thread panicked");
                    out.status(Status::Unavailable);
                }
            }
        });
    let thread = match spawned {
        Ok(thread) => Some(thread),
        Err(error) => {
            eprintln!("live transcript unavailable: cannot start its thread: {error}");
            None
        }
    };
    Handle { control, thread }
}

fn run<D: Decoder, E: Fn(&Update)>(
    timing: Timing,
    prepare: impl FnOnce() -> Result<D, String>,
    dir: &Path,
    control: &Control,
    out: &Publisher<E>,
) {
    out.status(Status::Preparing);
    let prepared = prepare();
    control.preparing.store(false, Ordering::SeqCst);
    let decoder = match prepared {
        Ok(decoder) => decoder,
        Err(message) => {
            eprintln!("live transcript unavailable: {message}");
            out.status(Status::Unavailable);
            return;
        }
    };
    // The recording may have ended, or the panel closed, while the model loaded.
    if control.phase() != Phase::Running {
        return;
    }

    let mut preview = Preview::new(decoder, dir);
    out.status(Status::Live);
    let mut failures = 0;
    let mut due = Instant::now() + timing.tick;
    loop {
        match control.wait_until(due) {
            Phase::Running => {}
            Phase::Finish => break,
            Phase::Cancel => return,
        }
        let started = Instant::now();
        let pass = preview.tick();
        if control.phase() == Phase::Cancel {
            return;
        }
        out.events(pass.events);
        match pass.error {
            Some(message) => {
                eprintln!("live transcript tick failed: {message}");
                failures += 1;
                if failures >= MAX_FAILURES {
                    out.status(Status::Unavailable);
                    return;
                }
            }
            None => failures = 0,
        }
        due = (started + timing.tick).max(Instant::now() + timing.min_rest);
    }

    let pass = preview.flush();
    if let Some(message) = &pass.error {
        eprintln!("live transcript final pass failed: {message}");
    }
    if control.phase() != Phase::Cancel {
        out.events(pass.events);
        out.status(Status::Finished);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::VecDeque;
    use std::io::Write;

    fn line(start_ms: i64, end_ms: i64, text: &str) -> Line {
        Line {
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    fn texts(lines: &[Line]) -> Vec<&str> {
        lines.iter().map(|l| l.text.as_str()).collect()
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("distil-live-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Append `samples` (int16) to a PCM file the way the sidecar does.
    fn append_pcm(path: &Path, samples: &[i16]) {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        for sample in samples {
            file.write_all(&sample.to_le_bytes()).unwrap();
        }
    }

    /// `seconds` of a constant-level tone that is clearly not silence.
    fn loud(seconds: usize) -> Vec<i16> {
        (0..seconds * 16_000)
            .map(|i| if i % 2 == 0 { 6_000 } else { -6_000 })
            .collect()
    }

    fn quiet(seconds: usize) -> Vec<i16> {
        vec![0; seconds * 16_000]
    }

    // ---- the commit split -------------------------------------------------

    #[test]
    fn a_line_before_the_horizon_commits_and_the_rest_stays_provisional() {
        // Window 10 s..30 s, so the horizon is 24 s.
        let lines = [
            line(11_000, 15_000, "one"),
            line(16_000, 23_000, "two"),
            line(25_000, 29_000, "three"),
        ];
        let split = split_commit(&lines, 10_000, 30_000, 0);
        assert_eq!(texts(&split.committed), ["one", "two"]);
        assert_eq!(texts(&split.provisional), ["three"]);
        assert_eq!(split.committed_ms, 23_000);
    }

    #[test]
    fn a_line_straddling_the_horizon_stays_provisional_whole() {
        let lines = [
            line(11_000, 15_000, "one"),
            line(20_000, 27_000, "straddles"),
            line(27_000, 29_000, "after"),
        ];
        let split = split_commit(&lines, 10_000, 30_000, 0);
        assert_eq!(texts(&split.committed), ["one"]);
        assert_eq!(texts(&split.provisional), ["straddles", "after"]);
        assert_eq!(split.committed_ms, 15_000);
    }

    #[test]
    fn a_line_ending_exactly_on_the_horizon_commits() {
        let split = split_commit(&[line(20_000, 24_000, "edge")], 10_000, 30_000, 0);
        assert_eq!(texts(&split.committed), ["edge"]);
    }

    #[test]
    fn committed_text_is_never_followed_by_older_text() {
        // Whisper overlapped two segments: the later-starting one ends first.
        let lines = [line(11_000, 26_000, "long"), line(12_000, 14_000, "inside")];
        let split = split_commit(&lines, 10_000, 30_000, 0);
        assert!(split.committed.is_empty());
        assert_eq!(texts(&split.provisional), ["long", "inside"]);
    }

    #[test]
    fn an_empty_window_or_no_lines_changes_nothing() {
        let empty_window = split_commit(&[line(0, 1_000, "x")], 5_000, 5_000, 800);
        assert_eq!(
            empty_window,
            Split {
                committed: vec![],
                provisional: vec![],
                committed_ms: 800
            }
        );
        let inverted_window = split_commit(&[line(0, 1_000, "x")], 6_000, 5_000, 800);
        assert_eq!(inverted_window.committed_ms, 800);
        assert!(inverted_window.committed.is_empty() && inverted_window.provisional.is_empty());
        let no_lines = split_commit(&[], 0, 20_000, 800);
        assert_eq!(
            no_lines,
            Split {
                committed: vec![],
                provisional: vec![],
                committed_ms: 800
            }
        );
    }

    #[test]
    fn a_window_shorter_than_the_horizon_commits_nothing() {
        let split = split_commit(&[line(0, 3_000, "hello")], 0, 5_000, 0);
        assert!(split.committed.is_empty());
        assert_eq!(texts(&split.provisional), ["hello"]);
    }

    #[test]
    fn the_watermark_never_moves_backwards() {
        // Already committed up to 40 s; this window only reaches 30 s.
        let split = split_commit(&[line(11_000, 15_000, "old")], 10_000, 30_000, 40_000);
        assert_eq!(split.committed_ms, 40_000);
        assert!(split.committed.is_empty() && split.provisional.is_empty());
        // And a fresh commit lifts it.
        let split = split_commit(&[line(50_000, 52_000, "new")], 40_000, 60_000, 40_000);
        assert_eq!(split.committed_ms, 52_000);
    }

    #[test]
    fn committed_ranges_are_never_emitted_twice() {
        let lines = [
            line(11_000, 15_000, "one"),
            line(16_000, 23_000, "two"),
            line(25_000, 29_000, "three"),
        ];
        let first = split_commit(&lines, 10_000, 30_000, 0);
        // The same audio decoded again on the next tick, 3 s later.
        let again = split_commit(&lines, 13_000, 33_000, first.committed_ms);
        assert!(
            again.committed.is_empty(),
            "re-emitted {:?}",
            again.committed
        );
        assert_eq!(texts(&again.provisional), ["three"]);
        assert_eq!(again.committed_ms, first.committed_ms);
    }

    #[test]
    fn a_resegmented_repeat_of_committed_text_is_dropped() {
        // "two" was committed at 16 s..23 s; whisper now cuts it 15.6 s..23.4 s.
        let repeat = [line(15_600, 23_400, "two"), line(25_000, 29_000, "three")];
        let split = split_commit(&repeat, 13_000, 33_000, 23_000);
        assert!(split.committed.is_empty());
        assert_eq!(texts(&split.provisional), ["three"]);
    }

    #[test]
    fn the_next_sentence_survives_timestamp_jitter_at_the_watermark() {
        // Committed up to 23 s; the next sentence starts 0.3 s early.
        let split = split_commit(&[line(22_700, 31_000, "next")], 13_000, 33_000, 23_000);
        assert_eq!(texts(&split.provisional), ["next"]);
    }

    #[test]
    fn times_are_clamped_to_the_window() {
        // Whisper placed a segment beyond the audio it was given.
        let split = split_commit(&[line(28_000, 45_000, "ghost")], 10_000, 30_000, 0);
        assert_eq!(split.provisional, [line(28_000, 30_000, "ghost")]);
    }

    #[test]
    fn a_flush_commits_everything_not_yet_committed() {
        let lines = [line(11_000, 15_000, "one"), line(25_000, 29_000, "tail")];
        let split = flush_commit(&lines, 10_000, 30_000, 12_000);
        // "one" has its middle (13 s) after the watermark, so it still counts.
        assert_eq!(texts(&split.committed), ["one", "tail"]);
        assert!(split.provisional.is_empty());
        assert_eq!(split.committed_ms, 29_000);
        let nothing_left = flush_commit(&lines, 10_000, 30_000, 29_000);
        assert!(nothing_left.committed.is_empty() && nothing_left.provisional.is_empty());
    }

    // ---- reading the PCM tail ----------------------------------------------

    #[test]
    fn the_tail_reader_follows_a_file_as_it_grows() {
        let dir = scratch_dir("reader");
        let path = dir.join("live.pcm");

        let first = read_pcm_since(&path, 0, 1_000);
        assert_eq!(
            first,
            PcmChunk {
                first_sample: 0,
                samples: vec![],
                next_offset: 0
            }
        );

        append_pcm(&path, &[0, 16_384, -16_384, i16::MIN]);
        let first = read_pcm_since(&path, 0, 1_000);
        assert_eq!(first.first_sample, 0);
        assert_eq!(first.samples, vec![0.0, 0.5, -0.5, -1.0]);
        assert_eq!(first.next_offset, 8);

        // Nothing new: same offset back.
        let idle = read_pcm_since(&path, first.next_offset, 1_000);
        assert!(idle.samples.is_empty());
        assert_eq!(idle.next_offset, 8);

        append_pcm(&path, &[8_192, 32_767]);
        let second = read_pcm_since(&path, first.next_offset, 1_000);
        assert_eq!(second.first_sample, 4);
        assert_eq!(second.samples[0], 0.25);
        assert!((second.samples[1] - 32_767.0 / 32_768.0).abs() < 1e-6);
        assert_eq!(second.next_offset, 12);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_trailing_odd_byte_waits_for_its_other_half() {
        let dir = scratch_dir("odd");
        let path = dir.join("live.pcm");
        let mut file = File::create(&path).unwrap();
        // One whole sample (0x4000 = 0.5) and the first byte of the next (0xC000 = -0.5).
        file.write_all(&[0x00, 0x40, 0x00]).unwrap();
        let chunk = read_pcm_since(&path, 0, 1_000);
        assert_eq!(chunk.samples, vec![0.5]);
        assert_eq!(chunk.next_offset, 2, "the odd byte must not be consumed");

        file.write_all(&[0xC0]).unwrap();
        let chunk = read_pcm_since(&path, chunk.next_offset, 1_000);
        assert_eq!(chunk.first_sample, 1);
        assert_eq!(chunk.samples, vec![-0.5]);
        assert_eq!(chunk.next_offset, 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_or_shortened_file_has_nothing_new() {
        let dir = scratch_dir("missing");
        let missing = read_pcm_since(&dir.join("nope.pcm"), 100, 1_000);
        assert!(missing.samples.is_empty());
        assert_eq!(missing.next_offset, 100);

        let path = dir.join("short.pcm");
        append_pcm(&path, &[1, 2]);
        let shortened = read_pcm_since(&path, 100, 1_000);
        assert!(shortened.samples.is_empty());
        assert_eq!(shortened.next_offset, 100);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_reader_far_behind_skips_to_the_newest_samples() {
        let dir = scratch_dir("behind");
        let path = dir.join("live.pcm");
        append_pcm(&path, &(0..100).collect::<Vec<i16>>());
        let chunk = read_pcm_since(&path, 0, 10);
        assert_eq!(chunk.first_sample, 90);
        assert_eq!(chunk.samples.len(), 10);
        assert_eq!(chunk.samples[0], 90.0 / 32768.0);
        assert_eq!(chunk.next_offset, 200);

        // With half a sample dangling at the end, the jump still lands on a
        // sample boundary.
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(&[0x55]).unwrap();
        let chunk = read_pcm_since(&path, 0, 10);
        assert_eq!(chunk.first_sample, 90);
        assert_eq!(chunk.samples[0], 90.0 / 32768.0);
        assert_eq!(chunk.next_offset, 200);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- the rolling window --------------------------------------------------

    #[test]
    fn the_window_keeps_only_the_newest_twenty_seconds() {
        let dir = scratch_dir("window");
        let path = dir.join("live.pcm");
        let mut tail = StreamTail::default();
        assert_eq!(tail.window_ms(), (0, 0));

        append_pcm(&path, &loud(5));
        tail.ingest(&path);
        assert_eq!(tail.window_ms(), (0, 5_000));
        assert_eq!(tail.offset(), 5 * 32_000);

        append_pcm(&path, &loud(30));
        tail.ingest(&path);
        assert_eq!(tail.window().len(), WINDOW_SAMPLES);
        assert_eq!(tail.window_ms(), (15_000, 35_000));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_reader_that_skipped_ahead_restarts_the_window_at_the_right_time() {
        let dir = scratch_dir("skip");
        let path = dir.join("live.pcm");
        append_pcm(&path, &loud(90));
        let mut tail = StreamTail::default();
        tail.ingest(&path);
        assert_eq!(tail.window_ms(), (70_000, 90_000));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- a tick, with a fake decoder -------------------------------------

    /// Hands back scripted answers and records what it was asked to decode.
    #[derive(Default)]
    struct Script {
        answers: VecDeque<Result<Vec<Line>, String>>,
        calls: Vec<usize>,
    }

    impl Decoder for Script {
        fn decode(&mut self, samples: &[f32]) -> Result<Vec<Line>, String> {
            self.calls.push(samples.len());
            self.answers.pop_front().unwrap_or_else(|| Ok(Vec::new()))
        }
    }

    fn preview_with(answers: Vec<Result<Vec<Line>, String>>, dir: &Path) -> Preview<Script> {
        Preview::new(
            Script {
                answers: answers.into(),
                calls: Vec::new(),
            },
            dir,
        )
    }

    #[test]
    fn silence_is_never_sent_to_the_decoder() {
        let dir = scratch_dir("silence");
        append_pcm(&dir.join(MIC_PCM), &quiet(10));
        append_pcm(&dir.join(SYSTEM_PCM), &quiet(10));
        let mut preview = preview_with(vec![], &dir);
        let pass = preview.tick();
        assert!(pass.events.is_empty() && pass.error.is_none());
        assert!(preview.decoder.calls.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_short_remainder_waits_for_more_audio_and_is_padded_for_the_last_pass() {
        let dir = scratch_dir("short");
        append_pcm(&dir.join(MIC_PCM), &loud(1)[..8_000]);
        let mut preview = preview_with(vec![Ok(vec![line(0, 400, "hi")])], &dir);
        let pass = preview.tick();
        assert!(pass.events.is_empty() && preview.decoder.calls.is_empty());
        // The recording ends there: whisper gets a full second, padded with silence.
        let last = preview.flush();
        assert_eq!(preview.decoder.calls, [16_000]);
        assert_eq!(texts(&last.events[0].committed), ["hi"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn whisper_is_given_only_the_audio_after_the_watermark() {
        let dir = scratch_dir("region");
        let mic = dir.join(MIC_PCM);
        append_pcm(&mic, &loud(12));
        // 12 s in, the horizon is at 6 s: "early" commits, "late" does not.
        let first = vec![line(0, 4_000, "early"), line(5_000, 11_000, "late")];
        let mut preview = preview_with(vec![Ok(first)], &dir);
        preview.tick();
        assert_eq!(preview.decoder.calls, [12 * 16_000]);
        assert_eq!(preview.channels[0].tail.committed_ms(), 4_000);

        // 3 s later whisper hears from 4 s on, not the whole window, and its
        // times are counted from there.
        append_pcm(&mic, &loud(3));
        preview
            .decoder
            .answers
            .push_back(Ok(vec![line(1_000, 7_500, "late news")]));
        let pass = preview.tick();
        assert_eq!(preview.decoder.calls, [12 * 16_000, 11 * 16_000]);
        assert!(pass.events[0].committed.is_empty());
        assert_eq!(
            pass.events[0].provisional,
            [line(5_000, 11_500, "late news")]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stream_with_speech_is_decoded_and_labelled() {
        let dir = scratch_dir("labels");
        append_pcm(&dir.join(MIC_PCM), &loud(8));
        append_pcm(&dir.join(SYSTEM_PCM), &quiet(8));
        let mut preview = preview_with(vec![Ok(vec![line(1_000, 6_000, "hello there")])], &dir);
        let pass = preview.tick();
        assert_eq!(
            preview.decoder.calls,
            [8 * 16_000],
            "only the mic window is decoded"
        );
        assert_eq!(pass.events.len(), 1);
        assert_eq!(pass.events[0].stream, Stream::Me);
        assert!(pass.events[0].committed.is_empty());
        assert_eq!(
            pass.events[0].provisional,
            [line(1_000, 6_000, "hello there")]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unchanged_tail_is_not_sent_again() {
        let dir = scratch_dir("unchanged");
        append_pcm(&dir.join(MIC_PCM), &loud(8));
        let same = || Ok(vec![line(1_000, 6_000, "hello there")]);
        let mut preview = preview_with(vec![same(), same()], &dir);
        assert_eq!(preview.tick().events.len(), 1);
        assert!(preview.tick().events.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lines_commit_as_the_window_moves_on_and_never_come_back() {
        let dir = scratch_dir("commit");
        let mic = dir.join(MIC_PCM);
        // Audio arrives 3 s at a time. The decoder hears the audio after the
        // watermark and answers with the sentences in it, timed from there, as
        // whisper would.
        let sentences = [
            (0, 4_000, "first"),
            (5_000, 9_000, "second"),
            (10_000, 14_000, "third"),
            (15_000, 19_000, "fourth"),
        ];
        let mut preview = preview_with(vec![], &dir);
        let mut committed_all: Vec<String> = Vec::new();
        let mut last_watermark = 0;
        for tick in 1..=10 {
            append_pcm(&mic, &loud(3));
            let (window_start, window_end) = {
                let end = tick * 3_000;
                ((end - WINDOW_MS).max(0), end)
            };
            let region_start = window_start.max(preview.channels[0].tail.committed_ms());
            let answer: Vec<Line> = sentences
                .iter()
                .filter(|(_, end, _)| *end <= window_end)
                .filter(|(start, _, _)| *start >= region_start)
                .map(|(start, end, text)| line(start - region_start, end - region_start, text))
                .collect();
            preview.decoder.answers.push_back(Ok(answer));
            let pass = preview.tick();
            for event in &pass.events {
                for committed in &event.committed {
                    committed_all.push(committed.text.clone());
                }
            }
            let watermark = preview.channels[0].tail.committed_ms();
            assert!(
                watermark >= last_watermark,
                "watermark went backwards at tick {tick}"
            );
            last_watermark = watermark;
        }
        assert_eq!(committed_all, ["first", "second", "third", "fourth"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn provisional_text_is_cleared_when_the_speech_has_gone_quiet() {
        let dir = scratch_dir("stale");
        let mic = dir.join(MIC_PCM);
        append_pcm(&mic, &loud(3));
        let mut preview = preview_with(vec![Ok(vec![line(0, 2_500, "maybe")])], &dir);
        let first = preview.tick();
        assert_eq!(texts(&first.events[0].provisional), ["maybe"]);

        // Whatever the text was backed by has slid out of the window, and
        // everything after the watermark is silent: nothing on screen is backed
        // by anything any more, and the decoder is left alone.
        append_pcm(&mic, &quiet(21));
        let pass = preview.tick();
        assert_eq!(pass.events.len(), 1);
        assert!(pass.events[0].committed.is_empty() && pass.events[0].provisional.is_empty());
        assert_eq!(
            preview.decoder.calls.len(),
            1,
            "the silent window was decoded"
        );
        // And it is only said once.
        append_pcm(&mic, &quiet(3));
        assert!(preview.tick().events.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lines_over_silent_audio_are_dropped_as_hallucinations() {
        let dir = scratch_dir("hallucination");
        let mic = dir.join(MIC_PCM);
        let mut samples = loud(4);
        samples.extend(quiet(4));
        append_pcm(&mic, &samples);
        let answer = vec![
            line(500, 3_500, "real words"),
            line(5_000, 7_500, "Thank you."),
        ];
        let mut preview = preview_with(vec![Ok(answer)], &dir);
        let pass = preview.tick();
        assert_eq!(texts(&pass.events[0].provisional), ["real words"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_decode_reports_the_error_and_sends_nothing() {
        let dir = scratch_dir("fail");
        append_pcm(&dir.join(MIC_PCM), &loud(8));
        let mut preview = preview_with(vec![Err("boom".into())], &dir);
        let pass = preview.tick();
        assert!(pass.events.is_empty());
        assert_eq!(pass.error.as_deref(), Some("boom"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_flush_commits_the_tail_including_the_latest_audio() {
        let dir = scratch_dir("flush");
        let mic = dir.join(MIC_PCM);
        append_pcm(&mic, &loud(8));
        let mut preview = preview_with(vec![Ok(vec![line(0, 6_000, "early words")])], &dir);
        let first = preview.tick();
        assert!(first.events[0].committed.is_empty());

        // Two more seconds arrive before the recording stops; the final pass
        // hears them too.
        append_pcm(&mic, &loud(2));
        preview.decoder.answers.push_back(Ok(vec![
            line(0, 6_000, "early words"),
            line(6_500, 9_800, "and the last bit"),
        ]));
        let last = preview.flush();
        assert_eq!(last.events.len(), 1);
        assert_eq!(
            texts(&last.events[0].committed),
            ["early words", "and the last bit"]
        );
        assert!(last.events[0].provisional.is_empty());
        assert_eq!(preview.channels[0].tail.committed_ms(), 9_800);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_flush_still_commits_what_was_on_show() {
        let dir = scratch_dir("flush-fail");
        append_pcm(&dir.join(MIC_PCM), &loud(8));
        let mut preview = preview_with(
            vec![Ok(vec![line(0, 6_000, "on show")]), Err("boom".into())],
            &dir,
        );
        preview.tick();
        let last = preview.flush();
        assert_eq!(last.error.as_deref(), Some("boom"));
        assert_eq!(texts(&last.events[0].committed), ["on show"]);
        assert!(last.events[0].provisional.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_flush_over_silence_commits_nothing_new() {
        let dir = scratch_dir("flush-quiet");
        append_pcm(&dir.join(MIC_PCM), &quiet(8));
        let mut preview = preview_with(vec![], &dir);
        let last = preview.flush();
        assert!(last.events.is_empty() && last.error.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- the snapshot ----------------------------------------------------

    #[test]
    fn the_snapshot_follows_the_events() {
        let mut snapshot = Snapshot::default();
        snapshot.apply(&Update::Status(Status::Live));
        snapshot.apply(&Update::Transcript(LiveEvent {
            stream: Stream::Me,
            committed: vec![line(0, 1_000, "a")],
            provisional: vec![line(1_000, 2_000, "b")],
        }));
        snapshot.apply(&Update::Transcript(LiveEvent {
            stream: Stream::Them,
            committed: vec![],
            provisional: vec![line(1_500, 2_500, "c")],
        }));
        snapshot.apply(&Update::Transcript(LiveEvent {
            stream: Stream::Me,
            committed: vec![line(1_000, 2_000, "b")],
            provisional: vec![],
        }));
        assert_eq!(snapshot.status, Status::Live);
        assert_eq!(snapshot.committed.len(), 2);
        assert_eq!(snapshot.provisional.len(), 1);
        assert_eq!(snapshot.provisional[0].stream, Stream::Them);

        snapshot.apply(&Update::Status(Status::Preparing));
        assert!(snapshot.committed.is_empty() && snapshot.provisional.is_empty());
    }

    #[test]
    fn events_serialise_the_way_the_panel_reads_them() {
        let event = LiveEvent {
            stream: Stream::Them,
            committed: vec![line(0, 1_000, "hi")],
            provisional: vec![],
        };
        assert_eq!(
            serde_json::to_value(&event).unwrap(),
            serde_json::json!({
                "stream": "them",
                "committed": [{ "startMs": 0, "endMs": 1000, "text": "hi" }],
                "provisional": [],
            })
        );
        let mut snapshot = Snapshot::default();
        snapshot.apply(&Update::Transcript(event));
        let json = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(json["status"], "preparing");
        assert_eq!(
            json["committed"][0],
            serde_json::json!({ "stream": "them", "startMs": 0, "endMs": 1000, "text": "hi" })
        );
    }

    // ---- the loop ----------------------------------------------------------

    fn fast() -> Timing {
        Timing {
            tick: Duration::from_millis(20),
            min_rest: Duration::from_millis(1),
        }
    }

    /// A decoder that always says the same thing, or always fails.
    struct Fixed(Result<Vec<Line>, String>);

    impl Decoder for Fixed {
        fn decode(&mut self, _samples: &[f32]) -> Result<Vec<Line>, String> {
            self.0.clone()
        }
    }

    type Seen = Arc<Mutex<Vec<Update>>>;

    fn start<D: Decoder + 'static>(
        dir: &Path,
        prepare: impl FnOnce() -> Result<D, String> + Send + 'static,
    ) -> (Handle, Arc<Mutex<Snapshot>>, Seen) {
        let snapshot = Arc::new(Mutex::new(Snapshot::default()));
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let handle = spawn_with(
            fast(),
            prepare,
            dir.to_path_buf(),
            snapshot.clone(),
            move |update| {
                sink.lock().unwrap().push(update.clone());
            },
        );
        (handle, snapshot, seen)
    }

    fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn statuses(seen: &Seen) -> Vec<Status> {
        seen.lock()
            .unwrap()
            .iter()
            .filter_map(|u| {
                if let Update::Status(s) = u {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect()
    }

    #[test]
    fn three_failed_ticks_in_a_row_stop_the_loop() {
        let dir = scratch_dir("loop-fail");
        append_pcm(&dir.join(MIC_PCM), &loud(8));
        let (handle, snapshot, seen) = start(&dir, || Ok(Fixed(Err("boom".into()))));
        wait_for("unavailable", || {
            lock(&snapshot).status == Status::Unavailable
        });
        assert_eq!(
            statuses(&seen),
            [Status::Preparing, Status::Live, Status::Unavailable]
        );
        handle.finish(Duration::from_secs(5));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failure_followed_by_a_success_resets_the_count() {
        struct Flaky(u32);
        impl Decoder for Flaky {
            fn decode(&mut self, _samples: &[f32]) -> Result<Vec<Line>, String> {
                // Fail twice, succeed, fail twice, succeed, ... never three in a row.
                self.0 += 1;
                if self.0 % 3 == 1 {
                    Ok(Vec::new())
                } else {
                    Err("flaky".into())
                }
            }
        }
        let dir = scratch_dir("loop-flaky");
        append_pcm(&dir.join(MIC_PCM), &loud(8));
        let (handle, snapshot, _seen) = start(&dir, || Ok(Flaky(0)));
        std::thread::sleep(Duration::from_millis(400));
        assert_eq!(
            lock(&snapshot).status,
            Status::Live,
            "gave up although no three ticks failed in a row"
        );
        handle.cancel();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_model_that_cannot_load_makes_the_preview_unavailable() {
        let dir = scratch_dir("loop-noload");
        let (handle, snapshot, seen) =
            start(&dir, || -> Result<Fixed, String> { Err("no model".into()) });
        wait_for("unavailable", || {
            lock(&snapshot).status == Status::Unavailable
        });
        assert_eq!(statuses(&seen), [Status::Preparing, Status::Unavailable]);
        handle.cancel();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn finishing_flushes_the_tail_and_ends_the_loop() {
        let dir = scratch_dir("loop-finish");
        append_pcm(&dir.join(MIC_PCM), &loud(8));
        let words = vec![line(500, 7_500, "last words")];
        let (handle, snapshot, seen) = start(&dir, move || Ok(Fixed(Ok(words))));
        wait_for("provisional text", || {
            !lock(&snapshot).provisional.is_empty()
        });
        assert!(lock(&snapshot).committed.is_empty());

        handle.finish(Duration::from_secs(5));
        let snapshot = lock(&snapshot).clone();
        assert_eq!(snapshot.status, Status::Finished);
        assert!(snapshot.provisional.is_empty());
        assert_eq!(snapshot.committed.len(), 1);
        assert_eq!(snapshot.committed[0].line.text, "last words");
        assert_eq!(statuses(&seen).last(), Some(&Status::Finished));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cancelling_stops_the_loop_without_a_flush() {
        let dir = scratch_dir("loop-cancel");
        append_pcm(&dir.join(MIC_PCM), &loud(8));
        let words = vec![line(500, 7_500, "unfinished")];
        let (handle, snapshot, seen) = start(&dir, move || Ok(Fixed(Ok(words))));
        wait_for("provisional text", || {
            !lock(&snapshot).provisional.is_empty()
        });
        handle.cancel();
        std::thread::sleep(Duration::from_millis(150));
        let count = seen.lock().unwrap().len();
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(
            seen.lock().unwrap().len(),
            count,
            "kept publishing after cancel"
        );
        assert!(lock(&snapshot).committed.is_empty());
        assert_ne!(lock(&snapshot).status, Status::Finished);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_canceller_stops_the_loop_for_good() {
        let dir = scratch_dir("loop-canceller");
        append_pcm(&dir.join(MIC_PCM), &loud(8));
        let words = vec![line(500, 7_500, "unfinished")];
        let (handle, snapshot, _seen) = start(&dir, move || Ok(Fixed(Ok(words))));
        wait_for("provisional text", || {
            !lock(&snapshot).provisional.is_empty()
        });
        handle.canceller().cancel();
        // Finishing afterwards cannot turn a cancel back into a flush.
        handle.finish(Duration::from_secs(5));
        let snapshot = lock(&snapshot).clone();
        assert_ne!(snapshot.status, Status::Finished);
        assert!(snapshot.committed.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn finish_does_not_wait_for_a_model_that_is_still_loading() {
        let dir = scratch_dir("loop-slow");
        let (handle, _snapshot, _seen) = start(&dir, || {
            std::thread::sleep(Duration::from_millis(600));
            Ok(Fixed(Ok(Vec::new())))
        });
        let started = Instant::now();
        handle.finish(Duration::from_secs(10));
        assert!(
            started.elapsed() < Duration::from_millis(300),
            "finish waited for the model"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn finish_gives_up_on_a_loop_that_will_not_stop() {
        struct Stuck;
        impl Decoder for Stuck {
            fn decode(&mut self, _samples: &[f32]) -> Result<Vec<Line>, String> {
                std::thread::sleep(Duration::from_millis(800));
                Ok(Vec::new())
            }
        }
        let dir = scratch_dir("loop-stuck");
        append_pcm(&dir.join(MIC_PCM), &loud(8));
        let (handle, _snapshot, _seen) = start(&dir, || Ok(Stuck));
        std::thread::sleep(Duration::from_millis(100));
        let started = Instant::now();
        handle.finish(Duration::from_millis(150));
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "finish waited too long"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_panicking_decoder_makes_the_preview_unavailable() {
        struct Panics;
        impl Decoder for Panics {
            fn decode(&mut self, _samples: &[f32]) -> Result<Vec<Line>, String> {
                panic!("decoder bug");
            }
        }
        let dir = scratch_dir("loop-panic");
        append_pcm(&dir.join(MIC_PCM), &loud(8));
        let (handle, snapshot, _seen) = start(&dir, || Ok(Panics));
        wait_for("unavailable", || {
            lock(&snapshot).status == Status::Unavailable
        });
        handle.cancel();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pcm_files_are_removed() {
        let dir = scratch_dir("remove");
        append_pcm(&dir.join(MIC_PCM), &[1]);
        append_pcm(&dir.join(SYSTEM_PCM), &[1]);
        std::fs::write(dir.join("audio-mic.wav"), b"x").unwrap();
        remove_pcm_files(&dir);
        assert!(!dir.join(MIC_PCM).exists() && !dir.join(SYSTEM_PCM).exists());
        assert!(dir.join("audio-mic.wav").exists(), "only the PCM files go");
        remove_pcm_files(&dir); // already gone: fine
        let _ = std::fs::remove_dir_all(&dir);
    }
}
