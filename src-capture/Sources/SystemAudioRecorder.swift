import AVFoundation
import CoreAudio
import Foundation

/// Captures system-output audio (everyone else in the meeting) with a Core
/// Audio process tap.
///
/// Deliberately *not* ScreenCaptureKit. SCK cannot capture audio without a
/// video configuration, so it forces the Screen & System Audio Recording
/// permission on someone who only ever wanted their meeting audio recorded —
/// the most alarming prompt on the list, in exchange for a 2x2 pixel video
/// stream that gets thrown away. A process tap captures the same mix and needs
/// no screen access at all, so Distil asks for microphone and calendar only.
///
/// Known limitation: the tap is bound to the default output device chosen when
/// recording starts. Switching output device mid-meeting stops system audio
/// from arriving; `ConvertingWavWriter.peakAmplitude` surfaces the resulting
/// silent track at the end of the recording.
final class SystemAudioRecorder {
    private let writer: ConvertingWavWriter
    private let sampleQueue = DispatchQueue(label: "system-audio-samples")

    private var tap: AudioTap?
    private var ioProcID: AudioDeviceIOProcID?
    private var running = false

    /// Called if capture dies underneath us. `nil` error means a clean external
    /// stop. Retained for parity with the session's stop lifecycle.
    var onStreamStopped: ((Error?) -> Void)?

    init(writer: ConvertingWavWriter) {
        self.writer = writer
    }

    func start() async throws {
        let tap = try AudioTap.create(name: "Distil system audio")
        self.tap = tap

        // Delivered on sampleQueue, which keeps the writer fed from one serial
        // context as ConvertingWavWriter requires.
        var proc: AudioDeviceIOProcID?
        do {
            try AudioTap.check(
                AudioDeviceCreateIOProcIDWithBlock(&proc, tap.aggregateID, sampleQueue) {
                    [weak self] _, inInputData, _, _, _ in
                    self?.append(inInputData)
                }, "install audio IO proc")
            ioProcID = proc
            try AudioTap.check(AudioDeviceStart(tap.aggregateID, proc), "start audio capture")
        } catch {
            await stop()
            throw error
        }
        running = true
    }

    func stop() async {
        guard let tap else { return }

        if running, let ioProcID {
            AudioDeviceStop(tap.aggregateID, ioProcID)
        }
        if let ioProcID {
            AudioDeviceDestroyIOProcID(tap.aggregateID, ioProcID)
        }
        tap.destroy()

        running = false
        ioProcID = nil
        self.tap = nil

        sampleQueue.sync {}  // drain in-flight callbacks before closing the file
        writer.finalize()
    }

    private func append(_ bufferList: UnsafePointer<AudioBufferList>) {
        guard let format = tap?.format,
            let buffer = AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: bufferList)
        else { return }
        writer.append(buffer)
    }
}
