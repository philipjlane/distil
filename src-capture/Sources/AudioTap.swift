import AVFoundation
import CoreAudio
import Foundation

/// A Core Audio process tap wrapped in a private aggregate device, ready for an
/// IOProc to be installed on it.
///
/// Shared by `SystemAudioRecorder`, which streams it to disk, and
/// `SystemAudioPermission`, which samples it briefly to work out whether macOS
/// has granted kTCCServiceAudioCapture — Core Audio exposes no API to ask.
struct AudioTap {
    let tapID: AudioObjectID
    let aggregateID: AudioObjectID
    let format: AVAudioFormat

    /// - Parameter excludingOwnProcess: leave `true` for capture so Distil never
    ///   records its own output. The permission probe passes `false` so a tone
    ///   played by this process would still be visible to the tap.
    static func create(name: String, excludingOwnProcess: Bool = true) throws -> AudioTap {
        let outputUID = try defaultOutputDeviceUID()

        let description = CATapDescription(
            __monoGlobalTapButExcludeProcesses: excludingOwnProcess ? ownProcessObjects() : [])
        description.name = name
        description.isPrivate = true

        var tap = AudioObjectID(kAudioObjectUnknown)
        try check(AudioHardwareCreateProcessTap(description, &tap), "create process tap")

        var asbd = try tapStreamFormat(tap)
        guard let format = AVAudioFormat(streamDescription: &asbd) else {
            AudioHardwareDestroyProcessTap(tap)
            throw CaptureError.message("unsupported system-audio tap format")
        }

        // The tap only produces samples once it is a member of a running
        // device, so wrap it in a private aggregate around the output device.
        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey: name,
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: try tapUID(tap),
                    kAudioSubTapDriftCompensationKey: true,
                ]
            ],
        ]
        var aggregateID = AudioObjectID(kAudioObjectUnknown)
        do {
            try check(
                AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &aggregateID),
                "create aggregate device")
        } catch {
            AudioHardwareDestroyProcessTap(tap)
            throw error
        }

        return AudioTap(tapID: tap, aggregateID: aggregateID, format: format)
    }

    func destroy() {
        if aggregateID != AudioObjectID(kAudioObjectUnknown) {
            AudioHardwareDestroyAggregateDevice(aggregateID)
        }
        if tapID != AudioObjectID(kAudioObjectUnknown) {
            AudioHardwareDestroyProcessTap(tapID)
        }
    }

    // MARK: Core Audio plumbing

    static func check(_ status: OSStatus, _ what: String) throws {
        guard status != noErr else { return }
        throw CaptureError.message("\(what) failed (OSStatus \(status))")
    }

    /// Our own audio process object, for exclusion from the tap. Empty if the
    /// lookup fails — a tap that records our own output is far better than no
    /// tap at all, and Distil emits no audio during a recording anyway.
    static func ownProcessObjects() -> [NSNumber] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var pid = getpid()
        var object = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address,
            UInt32(MemoryLayout<pid_t>.size), &pid, &size, &object)
        guard status == noErr, object != AudioObjectID(kAudioObjectUnknown) else { return [] }
        return [NSNumber(value: object)]
    }

    static func defaultOutputDeviceUID() throws -> String {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var device = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        try check(
            AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &device),
            "read default output device")
        guard device != AudioObjectID(kAudioObjectUnknown) else {
            throw CaptureError.message("no audio output device available")
        }
        return try stringProperty(device, kAudioDevicePropertyDeviceUID, "read output device UID")
    }

    /// Read a CFString-valued property.
    ///
    /// The destination is `Unmanaged`, not `CFString`: Core Audio writes a +1
    /// reference into this storage, and handing ARC a variable it believes
    /// already holds an object would have it release a string that was never
    /// there.
    static func stringProperty(
        _ object: AudioObjectID, _ selector: AudioObjectPropertySelector, _ what: String
    ) throws -> String {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        try check(AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value), what)
        guard let string = value?.takeRetainedValue() else {
            throw CaptureError.message("\(what) returned no value")
        }
        return string as String
    }

    static func tapStreamFormat(_ tap: AudioObjectID) throws -> AudioStreamBasicDescription {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var asbd = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        try check(
            AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &asbd), "read tap format")
        return asbd
    }

    static func tapUID(_ tap: AudioObjectID) throws -> String {
        try stringProperty(tap, kAudioTapPropertyUID, "read tap UID")
    }
}
