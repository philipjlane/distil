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

        // Apple voice processing (AEC): subtracts speaker playback from the
        // mic signal so meeting audio played on speakers doesn't bleed into
        // the Me stream. Ducking is minimised so the system-audio stream keeps
        // recording the meeting at full volume.
        do {
            try input.setVoiceProcessingEnabled(true)
            input.voiceProcessingOtherAudioDuckingConfiguration = .init(
                enableAdvancedDucking: false, duckingLevel: .min)
        } catch {
            Emit.event([
                "event": "warning",
                "message": "voice processing unavailable, recording raw mic: \(error.localizedDescription)",
            ])
        }

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
