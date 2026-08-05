import AVFoundation
import EventKit
import Foundation

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
