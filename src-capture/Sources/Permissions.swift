import AVFoundation
import EventKit
import Foundation

enum PermissionRequest {
    /// Ask macOS for one permission, then report the resulting state.
    ///
    /// This exists because `PermissionCheck` cannot do it: `authorizationStatus`
    /// only reads TCC, it never writes to it. An app that has merely *checked*
    /// stays `not-determined` and is absent from System Settings > Privacy &
    /// Security entirely — leaving no way to grant it by hand. The calls below
    /// are the ones that raise the prompt and register the app, so it shows up
    /// in Settings from then on.
    static func runAndPrint(kind: String) async {
        switch kind {
        case "microphone":
            _ = await AVCaptureDevice.requestAccess(for: .audio)
        case "calendar":
            // Held in a local: the request is async and the store must outlive it.
            let store = EKEventStore()
            _ = try? await store.requestFullAccessToEvents()
        default:
            Emit.error("unknown permission: \(kind)")
            exit(2)
        }
        PermissionCheck.runAndPrint()
    }
}

enum PermissionCheck {
    static func runAndPrint() {
        let microphone: String
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: microphone = "granted"
        case .notDetermined: microphone = "not-determined"
        case .denied: microphone = "denied"
        case .restricted: microphone = "restricted"
        @unknown default: microphone = "unknown"
        }
        let calendar: String
        switch EKEventStore.authorizationStatus(for: .event) {
        case .fullAccess: calendar = "granted"
        case .notDetermined: calendar = "not-determined"
        case .denied: calendar = "denied"
        case .restricted: calendar = "restricted"
        case .writeOnly: calendar = "write-only"
        @unknown default: calendar = "unknown"
        }
        // System-audio capture is deliberately absent: Core Audio taps are
        // gated by kTCCServiceAudioCapture, which has no public status API.
        // It prompts by itself the first time a tap is created.
        Emit.event([
            "event": "permissions",
            "microphone": microphone,
            "calendar": calendar,
        ])
    }
}
