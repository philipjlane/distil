import Foundation
import ScreenCaptureKit

/// Captures system-output audio (everyone else in the meeting) via an
/// audio-only ScreenCaptureKit stream. Requires the Screen & System Audio
/// Recording permission.
final class SystemAudioRecorder: NSObject, SCStreamOutput, SCStreamDelegate {
    private let writer: ConvertingWavWriter
    private let sampleQueue = DispatchQueue(label: "system-audio-samples")
    private var stream: SCStream?

    /// Called if the stream dies underneath us (display sleep, permission
    /// revoked mid-recording, …). `nil` error means a clean external stop.
    var onStreamStopped: ((Error?) -> Void)?

    init(writer: ConvertingWavWriter) {
        self.writer = writer
    }

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            throw CaptureError.message("no display available for system-audio capture")
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])

        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.excludesCurrentProcessAudio = true
        configuration.sampleRate = 16_000
        configuration.channelCount = 1
        // SCK requires a video configuration even for audio-only use; keep it
        // as cheap as possible and never register a .screen output.
        configuration.width = 2
        configuration.height = 2
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: sampleQueue)
        try await stream.startCapture()
        self.stream = stream
    }

    func stop() async {
        guard let stream else { return }
        self.stream = nil
        try? await stream.stopCapture()
        sampleQueue.sync {}  // drain in-flight sample callbacks before closing the file
        writer.finalize()
    }

    // MARK: SCStreamOutput

    func stream(
        _ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio, sampleBuffer.isValid else { return }
        writer.append(sampleBuffer)
    }

    // MARK: SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        guard self.stream != nil else { return }  // expected during our own stop()
        self.stream = nil
        onStreamStopped?(error)
    }
}
