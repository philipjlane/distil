import AVFoundation
import CoreGraphics
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
        let screen = CGPreflightScreenCaptureAccess() ? "granted" : "not-granted"
        let calendar: String
        switch EKEventStore.authorizationStatus(for: .event) {
        case .fullAccess: calendar = "granted"
        case .notDetermined: calendar = "not-determined"
        case .denied: calendar = "denied"
        case .restricted: calendar = "restricted"
        case .writeOnly: calendar = "write-only"
        @unknown default: calendar = "unknown"
        }
        Emit.event([
            "event": "permissions",
            "microphone": microphone,
            "screenRecording": screen,
            "calendar": calendar,
        ])
    }
}
