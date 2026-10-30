//! Call detection from microphone use.
//!
//! The capture sidecar's `watch-mic` mode reports, every couple of seconds,
//! which processes are capturing input. This module turns that heartbeat into
//! two signals: a call app has started using the microphone while nothing is
//! recording (offer to record), and the call has ended during a recording
//! (offer to stop). It is pure: the caller supplies the clock and the context,
//! so everything here is unit-tested without a Mac, a sidecar or Tauri.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use serde::Deserialize;

/// How long every call app seen during a recording must stay off the
/// microphone before the call counts as over. Long enough to ride out muting
/// and reconnects, short enough that the prompt arrives while it is still
/// relevant.
pub const ENDED_GRACE: Duration = Duration::from_secs(15);

/// A stdout line from the sidecar can in principle never end; past this many
/// buffered bytes the partial line is dropped rather than grown without bound.
const MAX_PENDING_LINE: usize = 1 << 20;

/// A process the sidecar reports as capturing input.
#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MicUser {
    pub bundle_id: String,
    pub pid: i32,
}

/// Bundle id prefix, display name and whether the app is a browser. Browsers
/// open the microphone for plenty that isn't a call (dictation, voice search),
/// so they only count when the calendar says a meeting is on.
const APPS: &[(&str, &str, bool)] = &[
    // Two ids: dotted-prefix matching keeps `com.microsoft.teams2` out of the
    // classic app's `com.microsoft.teams`.
    ("com.microsoft.teams2", "Teams", false),
    ("com.microsoft.teams", "Teams", false),
    ("us.zoom.xos", "Zoom", false),
    ("Cisco-Systems.Spark", "Webex", false),
    ("com.webex.meetingmanager", "Webex", false),
    ("com.tinyspeck.slackmacgap", "Slack", false),
    ("com.apple.FaceTime", "FaceTime", false),
    ("com.hnc.Discord", "Discord", false),
    ("com.google.Chrome", "Chrome", true),
    ("company.thebrowser.Browser", "Arc", true),
    ("com.microsoft.edgemac", "Edge", true),
    ("com.brave.Browser", "Brave", true),
    ("org.mozilla.firefox", "Firefox", true),
    // Safari captures through WebKit's shared GPU process, not its own.
    ("com.apple.Safari", "Safari", true),
    ("com.apple.WebKit.GPU", "Safari", true),
];

/// Whether `id` is `prefix` or one of its dotted children, ignoring ASCII case
/// (bundle ids are case-insensitive to LaunchServices).
fn has_bundle_prefix(id: &str, prefix: &str) -> bool {
    id.get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
        && matches!(id.as_bytes().get(prefix.len()), None | Some(b'.'))
}

/// The call app a bundle id belongs to, as (display name, is_browser).
///
/// Matches the app's own id and anything dotted beneath it, so helper processes
/// (`com.google.Chrome.helper`) count as the app, but a lookalike
/// (`com.microsoft.teamsfoo`) does not.
pub fn meeting_app(bundle_id: &str) -> Option<(&'static str, bool)> {
    APPS.iter()
        .find(|(prefix, ..)| has_bundle_prefix(bundle_id, prefix))
        .map(|(_, name, is_browser)| (*name, *is_browser))
}

#[derive(Debug, PartialEq)]
pub enum Signal {
    /// A qualifying call app began using the microphone and nothing is recording.
    Started { app: &'static str },
    /// Every call app seen during this recording has left the microphone.
    Ended { app: &'static str },
}

/// Turns microphone snapshots into [`Signal`]s.
///
/// Call [`update`](Detector::update) once per snapshot. The sidecar sends one
/// every tick, changed or not, so time-based rules only need the clock the
/// caller passes in; no timer of our own is involved.
#[derive(Default)]
pub struct Detector {
    /// Call apps in the previous snapshot, qualifying or not. An app has to
    /// appear in two consecutive snapshots to be confirmed, which debounces the
    /// brief microphone probes some apps make.
    seen: BTreeSet<(bool, &'static str)>,
    /// Confirmed apps that qualify as a call right now, as (is_browser, name).
    /// Sorted so native call apps come before browsers.
    active: BTreeSet<(bool, &'static str)>,
    /// During a recording: every call app confirmed on the microphone since it
    /// began, with the last time it was.
    on_call: BTreeMap<&'static str, Instant>,
    /// `Ended` has fired for the current recording.
    ended: bool,
}

impl Detector {
    /// Feed one snapshot of the processes capturing input.
    ///
    /// - An app is *confirmed active* once it appears in two consecutive
    ///   snapshots.
    /// - Browsers only qualify while `calendar_meeting` is true (a calendar
    ///   meeting is inside its prompt window); native call apps always do.
    /// - `Started` fires once each time a qualifying app goes from not active
    ///   to confirmed active while `recording` is false. Staying active, or
    ///   already being active when a recording ends, fires nothing.
    /// - While `recording`, every confirmed-active call app (browsers too,
    ///   whatever the calendar says) joins a per-recording set. `Ended` fires
    ///   once per recording, when that set is non-empty and no app in it has
    ///   been confirmed active for [`ENDED_GRACE`]. A recording that starts
    ///   with no call app on the microphone gets no `Ended` unless one appears
    ///   later. All per-recording state resets when `recording` goes false.
    pub fn update(
        &mut self,
        users: &[MicUser],
        recording: bool,
        calendar_meeting: bool,
        now: Instant,
    ) -> Option<Signal> {
        let in_snapshot: BTreeSet<(bool, &'static str)> = users
            .iter()
            .filter_map(|user| meeting_app(&user.bundle_id))
            .map(|(name, is_browser)| (is_browser, name))
            .collect();
        let confirmed: BTreeSet<_> = in_snapshot.intersection(&self.seen).copied().collect();
        self.seen = in_snapshot;

        let active: BTreeSet<_> = confirmed
            .iter()
            .filter(|(is_browser, _)| !is_browser || calendar_meeting)
            .copied()
            .collect();
        let started = active.difference(&self.active).next().copied();
        self.active = active;

        if !recording {
            self.on_call.clear();
            self.ended = false;
            return started.map(|(_, app)| Signal::Started { app });
        }

        for (_, app) in &confirmed {
            self.on_call.insert(*app, now);
        }
        if self.ended || self.on_call.is_empty() {
            return None;
        }
        let quiet = |last: &Instant| now.saturating_duration_since(*last) >= ENDED_GRACE;
        if !self.on_call.values().all(quiet) {
            return None;
        }
        self.ended = true;
        // Name the app that was last to leave.
        self.on_call
            .iter()
            .max_by_key(|(_, last)| **last)
            .map(|(app, _)| Signal::Ended { app })
    }

    /// The call app on the microphone and qualifying right now, for the tray
    /// row. Native apps win over browsers. Independent of `recording`: while a
    /// recording runs the call is still in progress, and the tray hides the row
    /// itself, so a recording stopped by hand mid-call offers to resume.
    pub fn current(&self) -> Option<&'static str> {
        self.active.iter().next().map(|(_, app)| *app)
    }
}

/// Parse a sidecar stdout line into the processes it reports, or `None` if it
/// is some other event or not JSON at all.
pub fn parse_mic_activity(line: &str) -> Option<Vec<MicUser>> {
    #[derive(Deserialize)]
    struct Report {
        event: String,
        #[serde(default)]
        processes: Vec<MicUser>,
    }
    let report: Report = serde_json::from_str(line).ok()?;
    (report.event == "mic-activity").then_some(report.processes)
}

/// Reassembles lines from a stream of chunks. Process stdout arrives in
/// arbitrary pieces: several lines at once, or a line split mid-way.
#[derive(Default)]
pub struct LineBuffer {
    pending: Vec<u8>,
}

impl LineBuffer {
    /// Add a chunk and return the lines it completed (without their newlines).
    /// A trailing partial line waits for the rest of it.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(chunk);
        let mut lines = Vec::new();
        while let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=end).collect();
            let text = String::from_utf8_lossy(&line[..end]);
            lines.push(text.trim_end_matches('\r').to_string());
        }
        if self.pending.len() > MAX_PENDING_LINE {
            self.pending.clear();
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(bundle_id: &str) -> MicUser {
        MicUser {
            bundle_id: bundle_id.into(),
            pid: 100,
        }
    }

    const TEAMS: &str = "com.microsoft.teams2";
    const ZOOM: &str = "us.zoom.xos";
    const CHROME: &str = "com.google.Chrome";

    /// Drives a `Detector` with a fake clock: each `tick` is one snapshot,
    /// `step` seconds after the last.
    struct Harness {
        detector: Detector,
        start: Instant,
        at: u64,
    }

    impl Harness {
        fn new() -> Self {
            Self {
                detector: Detector::default(),
                start: Instant::now(),
                at: 0,
            }
        }

        fn tick(
            &mut self,
            step: u64,
            ids: &[&str],
            recording: bool,
            calendar: bool,
        ) -> Option<Signal> {
            self.at += step;
            let users: Vec<MicUser> = ids.iter().map(|id| user(id)).collect();
            let now = self.start + Duration::from_secs(self.at);
            self.detector.update(&users, recording, calendar, now)
        }
    }

    #[test]
    fn known_apps_are_named_and_browsers_flagged() {
        assert_eq!(meeting_app("com.microsoft.teams2"), Some(("Teams", false)));
        assert_eq!(meeting_app("com.microsoft.teams"), Some(("Teams", false)));
        assert_eq!(meeting_app("us.zoom.xos"), Some(("Zoom", false)));
        assert_eq!(meeting_app("Cisco-Systems.Spark"), Some(("Webex", false)));
        assert_eq!(
            meeting_app("com.webex.meetingmanager"),
            Some(("Webex", false))
        );
        assert_eq!(
            meeting_app("com.tinyspeck.slackmacgap"),
            Some(("Slack", false))
        );
        assert_eq!(meeting_app("com.apple.FaceTime"), Some(("FaceTime", false)));
        assert_eq!(meeting_app("com.hnc.Discord"), Some(("Discord", false)));
        assert_eq!(meeting_app("com.google.Chrome"), Some(("Chrome", true)));
        assert_eq!(
            meeting_app("company.thebrowser.Browser"),
            Some(("Arc", true))
        );
        assert_eq!(meeting_app("com.microsoft.edgemac"), Some(("Edge", true)));
        assert_eq!(meeting_app("com.brave.Browser"), Some(("Brave", true)));
        assert_eq!(meeting_app("org.mozilla.firefox"), Some(("Firefox", true)));
        assert_eq!(meeting_app("com.apple.Safari"), Some(("Safari", true)));
        assert_eq!(meeting_app("com.apple.WebKit.GPU"), Some(("Safari", true)));
    }

    #[test]
    fn helper_processes_match_their_app() {
        assert_eq!(
            meeting_app("com.google.Chrome.helper"),
            Some(("Chrome", true))
        );
        assert_eq!(
            meeting_app("com.google.Chrome.helper.Renderer"),
            Some(("Chrome", true))
        );
        assert_eq!(
            meeting_app("com.microsoft.teams2.helper"),
            Some(("Teams", false))
        );
        assert_eq!(
            meeting_app("com.microsoft.teams.helper"),
            Some(("Teams", false))
        );
        assert_eq!(meeting_app("us.zoom.xos.helper"), Some(("Zoom", false)));
        assert_eq!(
            meeting_app("com.microsoft.edgemac.helper"),
            Some(("Edge", true))
        );
        assert_eq!(meeting_app("COM.GOOGLE.CHROME"), Some(("Chrome", true)));
    }

    #[test]
    fn lookalike_bundle_ids_do_not_match() {
        for id in [
            "",
            "com.microsoft.teamsfoo",
            "com.microsoft.team",
            "com.microsoft",
            "com.microsoft.Word",
            "com.google.ChromeRemoteDesktop",
            "us.zoom.xosx",
            "xus.zoom.xos",
            "com.apple.WebKit.WebContent",
            "com.spotify.client",
            "com.philipjlane.distil",
            "com.philipjlane.distil.capture",
            "é",
        ] {
            assert_eq!(meeting_app(id), None, "{id}");
        }
    }

    #[test]
    fn needs_two_consecutive_snapshots_to_confirm() {
        let mut h = Harness::new();
        assert_eq!(h.tick(2, &[TEAMS], false, false), None);
        assert_eq!(h.detector.current(), None);
        assert_eq!(
            h.tick(2, &[TEAMS], false, false),
            Some(Signal::Started { app: "Teams" })
        );
        assert_eq!(h.detector.current(), Some("Teams"));
    }

    #[test]
    fn a_brief_probe_never_starts_anything() {
        let mut h = Harness::new();
        assert_eq!(h.tick(2, &[TEAMS], false, false), None);
        assert_eq!(h.tick(2, &[], false, false), None);
        assert_eq!(h.tick(2, &[TEAMS], false, false), None);
        assert_eq!(h.tick(2, &[], false, false), None);
        assert_eq!(h.detector.current(), None);
    }

    #[test]
    fn started_fires_once_per_activation() {
        let mut h = Harness::new();
        h.tick(2, &[ZOOM], false, false);
        assert_eq!(
            h.tick(2, &[ZOOM], false, false),
            Some(Signal::Started { app: "Zoom" })
        );
        for _ in 0..10 {
            assert_eq!(h.tick(2, &[ZOOM], false, false), None);
        }
        // Leaving and coming back is a new activation.
        assert_eq!(h.tick(2, &[], false, false), None);
        assert_eq!(h.detector.current(), None);
        assert_eq!(h.tick(2, &[ZOOM], false, false), None);
        assert_eq!(
            h.tick(2, &[ZOOM], false, false),
            Some(Signal::Started { app: "Zoom" })
        );
    }

    #[test]
    fn started_when_a_second_app_joins_but_not_for_the_first_again() {
        let mut h = Harness::new();
        h.tick(2, &[ZOOM], false, false);
        assert_eq!(
            h.tick(2, &[ZOOM], false, false),
            Some(Signal::Started { app: "Zoom" })
        );
        h.tick(2, &[ZOOM, TEAMS], false, false);
        assert_eq!(
            h.tick(2, &[ZOOM, TEAMS], false, false),
            Some(Signal::Started { app: "Teams" })
        );
        assert_eq!(h.tick(2, &[ZOOM, TEAMS], false, false), None);
    }

    #[test]
    fn processes_of_one_app_count_as_one() {
        let mut h = Harness::new();
        h.tick(
            2,
            &["com.microsoft.teams2", "com.microsoft.teams2.helper"],
            false,
            false,
        );
        // Different helper processes on the next tick still confirm the app.
        assert_eq!(
            h.tick(2, &["com.microsoft.teams2.helper"], false, false),
            Some(Signal::Started { app: "Teams" })
        );
        assert_eq!(
            h.tick(
                2,
                &["com.microsoft.teams2", "com.microsoft.teams"],
                false,
                false
            ),
            None
        );
    }

    #[test]
    fn unrelated_microphone_users_are_ignored() {
        let mut h = Harness::new();
        for _ in 0..5 {
            assert_eq!(
                h.tick(
                    2,
                    &["com.apple.voicememos", "com.spotify.client"],
                    false,
                    true
                ),
                None
            );
        }
        assert_eq!(h.detector.current(), None);
    }

    #[test]
    fn browsers_need_a_calendar_meeting() {
        let mut h = Harness::new();
        for _ in 0..5 {
            assert_eq!(h.tick(2, &[CHROME], false, false), None);
        }
        assert_eq!(h.detector.current(), None);

        let mut h = Harness::new();
        h.tick(2, &[CHROME], false, true);
        assert_eq!(
            h.tick(2, &[CHROME], false, true),
            Some(Signal::Started { app: "Chrome" })
        );
        assert_eq!(h.detector.current(), Some("Chrome"));
        assert_eq!(h.tick(2, &[CHROME], false, true), None);
    }

    #[test]
    fn a_browser_already_on_the_mic_starts_when_the_calendar_window_opens() {
        let mut h = Harness::new();
        h.tick(2, &[CHROME], false, false);
        assert_eq!(h.tick(2, &[CHROME], false, false), None);
        assert_eq!(
            h.tick(2, &[CHROME], false, true),
            Some(Signal::Started { app: "Chrome" })
        );
        // Once the window closes the browser stops qualifying.
        assert_eq!(h.tick(2, &[CHROME], false, false), None);
        assert_eq!(h.detector.current(), None);
    }

    #[test]
    fn native_apps_win_over_browsers_for_the_tray() {
        let mut h = Harness::new();
        h.tick(2, &[CHROME, TEAMS], false, true);
        assert!(h.tick(2, &[CHROME, TEAMS], false, true).is_some());
        assert_eq!(h.detector.current(), Some("Teams"));
    }

    #[test]
    fn nothing_starts_while_recording() {
        let mut h = Harness::new();
        h.tick(2, &[TEAMS], true, false);
        assert_eq!(h.tick(2, &[TEAMS], true, false), None);
        assert_eq!(h.detector.current(), Some("Teams"));
    }

    #[test]
    fn stopping_a_recording_mid_call_does_not_announce_the_call_again() {
        let mut h = Harness::new();
        h.tick(2, &[TEAMS], false, false);
        assert!(h.tick(2, &[TEAMS], false, false).is_some());
        h.tick(2, &[TEAMS], true, false);
        h.tick(2, &[TEAMS], true, false);
        assert_eq!(h.tick(2, &[TEAMS], false, false), None);
        assert_eq!(h.tick(2, &[TEAMS], false, false), None);
        assert_eq!(h.detector.current(), Some("Teams"));
    }

    #[test]
    fn ended_fires_after_the_grace_period() {
        let mut h = Harness::new();
        h.tick(2, &[TEAMS], true, false);
        // Confirmed here, so this is the last moment Teams was on the mic.
        assert_eq!(h.tick(2, &[TEAMS], true, false), None);
        assert_eq!(h.tick(2, &[], true, false), None);
        // 14 s later is still inside the grace.
        assert_eq!(h.tick(12, &[], true, false), None);
        // Exactly 15 s.
        assert_eq!(
            h.tick(1, &[], true, false),
            Some(Signal::Ended { app: "Teams" })
        );
    }

    #[test]
    fn no_ended_before_the_grace_period() {
        let mut h = Harness::new();
        h.tick(2, &[ZOOM], true, false);
        h.tick(2, &[ZOOM], true, false);
        // Seven quiet ticks is 14 s since Zoom was last on the mic.
        for _ in 0..7 {
            assert_eq!(h.tick(2, &[], true, false), None);
        }
    }

    #[test]
    fn coming_back_within_the_grace_cancels_the_end() {
        let mut h = Harness::new();
        h.tick(2, &[TEAMS], true, false);
        h.tick(2, &[TEAMS], true, false);
        assert_eq!(h.tick(10, &[], true, false), None);
        // Muting dropped the mic for a few ticks; two snapshots reconfirm it.
        assert_eq!(h.tick(2, &[TEAMS], true, false), None);
        assert_eq!(h.tick(2, &[TEAMS], true, false), None);
        assert_eq!(h.tick(14, &[], true, false), None);
        assert_eq!(
            h.tick(1, &[], true, false),
            Some(Signal::Ended { app: "Teams" })
        );
    }

    #[test]
    fn ended_fires_only_once_per_recording() {
        let mut h = Harness::new();
        h.tick(2, &[TEAMS], true, false);
        h.tick(2, &[TEAMS], true, false);
        assert!(h.tick(20, &[], true, false).is_some());
        for _ in 0..20 {
            assert_eq!(h.tick(2, &[], true, false), None);
        }
        // Not even if another call follows in the same recording.
        h.tick(2, &[ZOOM], true, false);
        h.tick(2, &[ZOOM], true, false);
        assert_eq!(h.tick(30, &[], true, false), None);
    }

    #[test]
    fn no_ended_without_a_call_app() {
        let mut h = Harness::new();
        for _ in 0..60 {
            assert_eq!(h.tick(2, &["com.apple.voicememos"], true, false), None);
        }
        // An unconfirmed blip is not a call either.
        assert_eq!(h.tick(2, &[TEAMS], true, false), None);
        assert_eq!(h.tick(2, &[], true, false), None);
        assert_eq!(h.tick(60, &[], true, false), None);
    }

    #[test]
    fn a_call_app_appearing_mid_recording_arms_the_end() {
        let mut h = Harness::new();
        for _ in 0..5 {
            assert_eq!(h.tick(2, &[], true, false), None);
        }
        h.tick(2, &[ZOOM], true, false);
        assert_eq!(h.tick(2, &[ZOOM], true, false), None);
        assert_eq!(
            h.tick(30, &[], true, false),
            Some(Signal::Ended { app: "Zoom" })
        );
    }

    #[test]
    fn the_recording_waits_for_every_call_app() {
        let mut h = Harness::new();
        h.tick(2, &[TEAMS, ZOOM], true, false);
        h.tick(2, &[TEAMS, ZOOM], true, false);
        // Teams leaves; Zoom keeps the call alive.
        for _ in 0..10 {
            assert_eq!(h.tick(2, &[ZOOM], true, false), None);
        }
        // Zoom leaves last, so it is the one named, once its own grace has run.
        assert_eq!(h.tick(2, &[], true, false), None);
        assert_eq!(
            h.tick(13, &[], true, false),
            Some(Signal::Ended { app: "Zoom" })
        );
    }

    #[test]
    fn browsers_arm_the_end_even_without_a_calendar_meeting() {
        let mut h = Harness::new();
        h.tick(2, &[CHROME], true, false);
        assert_eq!(h.tick(2, &[CHROME], true, false), None);
        assert_eq!(
            h.tick(20, &[], true, false),
            Some(Signal::Ended { app: "Chrome" })
        );
    }

    #[test]
    fn a_new_recording_re_arms_the_end() {
        let mut h = Harness::new();
        h.tick(2, &[TEAMS], true, false);
        h.tick(2, &[TEAMS], true, false);
        assert!(h.tick(20, &[], true, false).is_some());
        assert_eq!(h.tick(2, &[], true, false), None);

        // Recording stops, then a later one starts with a fresh call.
        assert_eq!(h.tick(2, &[], false, false), None);
        h.tick(2, &[ZOOM], false, false);
        assert_eq!(
            h.tick(2, &[ZOOM], false, false),
            Some(Signal::Started { app: "Zoom" })
        );
        h.tick(2, &[ZOOM], true, false);
        h.tick(2, &[ZOOM], true, false);
        assert_eq!(
            h.tick(20, &[], true, false),
            Some(Signal::Ended { app: "Zoom" })
        );
    }

    #[test]
    fn per_recording_state_does_not_leak_into_the_next_recording() {
        let mut h = Harness::new();
        h.tick(2, &[TEAMS], true, false);
        h.tick(2, &[TEAMS], true, false);
        // The recording is stopped by hand before the call ends...
        assert_eq!(h.tick(2, &[], false, false), None);
        // ...so a long-quiet follow-up recording must not inherit its Teams.
        assert_eq!(h.tick(2, &[], true, false), None);
        assert_eq!(h.tick(60, &[], true, false), None);
    }

    #[test]
    fn empty_snapshots_are_harmless() {
        let mut h = Harness::new();
        for recording in [false, true, false] {
            assert_eq!(h.tick(2, &[], recording, true), None);
        }
        assert_eq!(h.detector.current(), None);
    }

    #[test]
    fn mic_activity_lines_parse() {
        let line = r#"{"event":"mic-activity","processes":[{"bundleId":"us.zoom.xos","pid":4242},{"bundleId":"com.google.Chrome.helper","pid":7}]}"#;
        assert_eq!(
            parse_mic_activity(line),
            Some(vec![
                MicUser {
                    bundle_id: "us.zoom.xos".into(),
                    pid: 4242
                },
                MicUser {
                    bundle_id: "com.google.Chrome.helper".into(),
                    pid: 7
                },
            ])
        );
        assert_eq!(
            parse_mic_activity(r#"{"event":"mic-activity","processes":[]}"#),
            Some(vec![])
        );
        assert_eq!(
            parse_mic_activity(r#"{"event":"mic-activity"}"#),
            Some(vec![])
        );
    }

    #[test]
    fn other_lines_are_not_mic_activity() {
        assert_eq!(
            parse_mic_activity(r#"{"event":"error","message":"boom"}"#),
            None
        );
        assert_eq!(
            parse_mic_activity(r#"{"event":"calendar","events":[]}"#),
            None
        );
        assert_eq!(parse_mic_activity("not json"), None);
        assert_eq!(parse_mic_activity(""), None);
    }

    #[test]
    fn lines_are_reassembled_across_chunks() {
        let mut buffer = LineBuffer::default();
        assert!(buffer.push(b"{\"event\":\"mic-").is_empty());
        assert_eq!(
            buffer.push(b"activity\"}\n{\"ev"),
            vec![r#"{"event":"mic-activity"}"#]
        );
        assert_eq!(
            buffer.push(b"ent\":\"b\"}\n\nlast\r\n"),
            vec![r#"{"event":"b"}"#, "", "last"]
        );
        assert!(buffer.push(b"").is_empty());
    }

    #[test]
    fn several_lines_in_one_chunk_all_come_out() {
        let mut buffer = LineBuffer::default();
        assert_eq!(buffer.push(b"a\nb\nc\nd"), vec!["a", "b", "c"]);
        assert_eq!(buffer.push(b"\n"), vec!["d"]);
    }

    #[test]
    fn multibyte_characters_split_across_chunks_survive() {
        let bytes = "caf\u{e9}\n".as_bytes();
        let mut buffer = LineBuffer::default();
        assert!(buffer.push(&bytes[..4]).is_empty());
        assert_eq!(buffer.push(&bytes[4..]), vec!["caf\u{e9}"]);
    }

    #[test]
    fn a_runaway_line_is_dropped_not_hoarded() {
        let mut buffer = LineBuffer::default();
        assert!(buffer.push(&vec![b'x'; MAX_PENDING_LINE + 1]).is_empty());
        assert_eq!(buffer.push(b"ok\n"), vec!["ok"]);
    }
}
