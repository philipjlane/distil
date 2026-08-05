import AVFoundation
import CoreAudio
import Foundation

/// Works out whether macOS has granted kTCCServiceAudioCapture.
///
/// Core Audio has no authorization API — the tap properties are UID,
/// description and format, and nothing else. A denied tap is not an error
/// either: it runs normally and delivers frames of pure silence. So the only
/// way to know is to listen and see whether anything arrives.
///
/// Ambient audio is not a usable reference. `kAudioProcessPropertyIsRunningOutput`
/// reports that a process holds a running output IOProc, not that sound is
/// audible — a paused Spotify sets it — so silence from the tap could mean
/// "denied" or "nothing was playing", and the two are indistinguishable.
///
/// Instead the probe plays its own brief, quiet tone and listens for it. That
/// removes the ambiguity entirely: the reference signal is known to exist, so
/// silence can only mean the tap was denied.
///
/// Creating the tap is also what raises the macOS prompt in the first place, so
/// this doubles as the "request" for system audio.
enum SystemAudioPermission {
    /// Quiet and brief — the user pressed a button, a faint blip is acceptable,
    /// and reliability matters more than being inaudible. A near-ultrasonic
    /// tone would be politer but some output devices roll it off, which would
    /// read as a denial.
    private static let toneFrequency = 1_000.0
    private static let toneAmplitude: Float = 0.02
    private static let toneSeconds = 0.25
    /// Comfortably above dither and denormal noise, far below the tone.
    private static let audibleThreshold: Float = 0.0005

    static func probeAndPrint() {
        Emit.event(["event": "systemAudio", "status": probe()])
    }

    static func probe() -> String {
        let tap: AudioTap
        do {
            // Must NOT exclude our own process — our tone is the reference.
            tap = try AudioTap.create(name: "Distil audio check", excludingOwnProcess: false)
        } catch {
            return "unavailable"
        }
        defer { tap.destroy() }

        let queue = DispatchQueue(label: "system-audio-probe")
        let peak = Peak()

        var proc: AudioDeviceIOProcID?
        guard
            AudioDeviceCreateIOProcIDWithBlock(&proc, tap.aggregateID, queue, {
                _, input, _, _, _ in
                guard let buffer = AVAudioPCMBuffer(pcmFormat: tap.format, bufferListNoCopy: input),
                    let channel = buffer.floatChannelData
                else { return }
                var frame = Float(0)
                for i in 0..<Int(buffer.frameLength) { frame = max(frame, abs(channel[0][i])) }
                peak.observe(frame)
            }) == noErr, let proc
        else { return "unavailable" }
        defer { AudioDeviceDestroyIOProcID(tap.aggregateID, proc) }

        guard AudioDeviceStart(tap.aggregateID, proc) == noErr else { return "unavailable" }
        playTone()
        // Let the tail of the tone travel through the tap before deciding.
        Thread.sleep(forTimeInterval: 0.2)
        AudioDeviceStop(tap.aggregateID, proc)
        queue.sync {}  // let in-flight callbacks land before reading the peak

        return peak.value > audibleThreshold ? "granted" : "denied"
    }

    /// Play the reference tone, blocking until it has finished.
    private static func playTone() {
        let engine = AVAudioEngine()
        guard
            let format = AVAudioFormat(
                standardFormatWithSampleRate: engine.outputNode.outputFormat(forBus: 0).sampleRate,
                channels: 1)
        else { return }

        var phase = 0.0
        let increment = 2.0 * Double.pi * toneFrequency / format.sampleRate
        let source = AVAudioSourceNode(format: format) { _, _, frameCount, audioBufferList in
            let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
            for frame in 0..<Int(frameCount) {
                let value = Float(sin(phase)) * toneAmplitude
                phase += increment
                if phase > 2.0 * Double.pi { phase -= 2.0 * Double.pi }
                for buffer in buffers {
                    buffer.mData?.assumingMemoryBound(to: Float.self)[frame] = value
                }
            }
            return noErr
        }

        engine.attach(source)
        engine.connect(source, to: engine.mainMixerNode, format: format)
        do {
            try engine.start()
        } catch {
            return
        }
        Thread.sleep(forTimeInterval: toneSeconds)
        engine.stop()
    }

    /// Peak shared between the IO thread and the caller.
    private final class Peak {
        private let lock = NSLock()
        private var peak: Float = 0

        func observe(_ sample: Float) {
            lock.lock()
            peak = max(peak, sample)
            lock.unlock()
        }

        var value: Float {
            lock.lock()
            defer { lock.unlock() }
            return peak
        }
    }
}
