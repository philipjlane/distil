import AVFoundation
import Foundation

/// Captures the default input device (your voice) via AVAudioEngine.
final class MicRecorder {
    private let engine = AVAudioEngine()
    private let writer: ConvertingWavWriter

    init(writer: ConvertingWavWriter) {
        self.writer = writer
    }

    static func requestAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .audio)
        default:
            return false
        }
    }

    func start() throws {
        let input = engine.inputNode

        // Do NOT enable setVoiceProcessingEnabled here: in this headless
        // sidecar it produced an all-zero mic stream (macOS 26, 2026-08-03).
        // Speaker bleed is handled by the merge-time filter instead.

        engine.prepare()
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw CaptureError.message("microphone reports an invalid format")
        }
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [writer] buffer, _ in
            writer.append(buffer)
        }
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        writer.finalize()
    }
}
