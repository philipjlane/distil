import AVFoundation
import CoreGraphics
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
        let screen = CGPreflightScreenCaptureAccess() ? "granted" : "not-granted"
        Emit.event([
            "event": "permissions",
            "microphone": microphone,
            "screenRecording": screen,
        ])
    }
}
