import CoreAudio
import Foundation

/// Reports which processes are capturing microphone input, so the host can tell
/// when a call starts or ends without being told.
///
/// Reads Core Audio's process objects (macOS 14.0+; the package floor is 14.2)
/// and asks each one whether it is running input. That is metadata about other
/// processes, not audio: nothing is opened, tapped or recorded, so it should
/// need no TCC permission (unlike the mic recorder and the system-audio tap).
///
/// One line is emitted on every tick, not only when the list changes. The host
/// treats the stream as a heartbeat for its time-based logic (the grace period
/// before a call counts as over), so a quiet tick still has to be said aloud.
enum MicActivityWatcher {
    private static let pollInterval = DispatchTimeInterval.seconds(2)
    /// Held for the life of the process; a dropped timer source silently stops.
    private static var timer: DispatchSourceTimer?

    static func run() {
        StopTriggers.install { exit(0) }

        let source = DispatchSource.makeTimerSource(queue: .main)
        source.schedule(deadline: .now(), repeating: pollInterval)
        source.setEventHandler { emitSnapshot() }
        source.resume()
        timer = source
    }

    private static func emitSnapshot() {
        let processes = inputUsers().map { user -> [String: Any] in
            ["bundleId": user.bundleID, "pid": Int(user.pid)]
        }
        Emit.event(["event": "mic-activity", "processes": processes])
    }

    /// Processes with audio input running right now, excluding ourselves. An
    /// object whose properties cannot be read (it may exit mid-poll) is skipped.
    private static func inputUsers() -> [(bundleID: String, pid: pid_t)] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyProcessObjectList,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        let system = AudioObjectID(kAudioObjectSystemObject)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(system, &address, 0, nil, &size) == noErr, size > 0
        else { return [] }

        var objects = [AudioObjectID](
            repeating: AudioObjectID(kAudioObjectUnknown),
            count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(system, &address, 0, nil, &size, &objects) == noErr
        else { return [] }
        // The list can shrink between the size query and the read.
        objects = Array(objects.prefix(Int(size) / MemoryLayout<AudioObjectID>.size))

        var users: [(bundleID: String, pid: pid_t)] = []
        for object in objects where object != AudioObjectID(kAudioObjectUnknown) {
            guard let running: UInt32 = scalar(object, kAudioProcessPropertyIsRunningInput),
                running != 0,
                let pid: pid_t = scalar(object, kAudioProcessPropertyPID),
                pid != getpid(),
                let bundleID = try? AudioTap.stringProperty(
                    object, kAudioProcessPropertyBundleID, "read process bundle ID"),
                !bundleID.isEmpty
            else { continue }
            users.append((bundleID: bundleID, pid: pid))
        }
        return users
    }

    /// Read a fixed-size integer property, or nil if the object won't say.
    private static func scalar<T: FixedWidthInteger>(
        _ object: AudioObjectID, _ selector: AudioObjectPropertySelector
    ) -> T? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var value: T = 0
        var size = UInt32(MemoryLayout<T>.size)
        guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value) == noErr
        else { return nil }
        return value
    }
}
