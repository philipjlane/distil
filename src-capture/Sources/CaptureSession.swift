import Foundation

enum CaptureError: Error, LocalizedError {
    case message(String)

    var errorDescription: String? {
        if case let .message(text) = self { return text }
        return nil
    }
}

/// Owns both recorders and the stop lifecycle. Emits JSON events on stdout for
/// the host app: `started`, `error`, `stopped`.
final class CaptureSession {
    private let outDir: URL
    private let micWriter: ConvertingWavWriter
    private let systemWriter: ConvertingWavWriter
    private let mic: MicRecorder
    private let system: SystemAudioRecorder
    private var stopping = false
    private let stateQueue = DispatchQueue(label: "capture-state")

    init(outDir: URL) throws {
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        self.outDir = outDir
        micWriter = try ConvertingWavWriter(url: outDir.appendingPathComponent("audio-mic.wav"))
        systemWriter = try ConvertingWavWriter(url: outDir.appendingPathComponent("audio-system.wav"))
        mic = MicRecorder(writer: micWriter)
        system = SystemAudioRecorder(writer: systemWriter)
    }

    func start() {
        system.onStreamStopped = { [weak self] error in
            if let error {
                Emit.error("system audio stream stopped: \(error.localizedDescription)")
            }
            self?.requestStop(exitCode: error == nil ? 0 : 1)
        }
        Task {
            guard await MicRecorder.requestAccess() else {
                Emit.error("microphone access denied — grant it in System Settings → Privacy & Security → Microphone")
                exit(1)
            }
            do {
                try await system.start()
            } catch {
                Emit.error("system audio capture failed: \(error.localizedDescription) — grant Screen & System Audio Recording in System Settings → Privacy & Security")
                exit(1)
            }
            do {
                try mic.start()
            } catch {
                Emit.error("microphone capture failed: \(error.localizedDescription)")
                await system.stop()
                exit(1)
            }
            Emit.event(["event": "started", "dir": outDir.path])
        }
    }

    func requestStop(exitCode: Int32 = 0) {
        let alreadyStopping = stateQueue.sync { () -> Bool in
            if stopping { return true }
            stopping = true
            return false
        }
        guard !alreadyStopping else { return }

        Task {
            mic.stop()
            await system.stop()
            Emit.event([
                "event": "stopped",
                "micSeconds": micWriter.secondsWritten,
                "systemSeconds": systemWriter.secondsWritten,
            ])
            exit(exitCode)
        }
    }
}
