import AVFoundation
import CoreAudio
import Foundation

/// Captures system-output audio (everyone else in the meeting) with a Core
/// Audio process tap feeding a private aggregate device.
///
/// Deliberately *not* ScreenCaptureKit. SCK cannot capture audio without a
/// video configuration, so it forces the Screen & System Audio Recording
/// permission on someone who only ever wanted their meeting audio recorded —
/// the most alarming prompt on the list, in exchange for a 2x2 pixel video
/// stream that gets thrown away. A process tap captures the same mix and needs
/// no screen access at all, so Distil asks for microphone and calendar only.
///
/// Known limitation: the tap is bound to the default output device chosen when
/// recording starts. Switching output device mid-meeting stops system audio
/// from arriving; `ConvertingWavWriter.peakAmplitude` surfaces the resulting
/// silent track at the end of the recording.
final class SystemAudioRecorder {
    private let writer: ConvertingWavWriter
    private let sampleQueue = DispatchQueue(label: "system-audio-samples")

    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var ioProcID: AudioDeviceIOProcID?
    private var sourceFormat: AVAudioFormat?
    private var running = false

    /// Called if capture dies underneath us. `nil` error means a clean external
    /// stop. Retained for parity with the session's stop lifecycle.
    var onStreamStopped: ((Error?) -> Void)?

    init(writer: ConvertingWavWriter) {
        self.writer = writer
    }

    func start() async throws {
        let outputUID = try Self.defaultOutputDeviceUID()

        // Exclude our own audio so Distil can never record its own playback.
        let description = CATapDescription(
            __monoGlobalTapButExcludeProcesses: Self.ownProcessObjects())
        description.name = "Distil system audio"
        description.isPrivate = true

        var tap = AudioObjectID(kAudioObjectUnknown)
        try Self.check(AudioHardwareCreateProcessTap(description, &tap), "create process tap")
        tapID = tap

        var asbd = try Self.tapStreamFormat(tap)
        guard let format = AVAudioFormat(streamDescription: &asbd) else {
            throw CaptureError.message("unsupported system-audio tap format")
        }
        sourceFormat = format

        // The tap only produces samples once it is a member of a running
        // device, so wrap it in a private aggregate around the output device.
        let tapUID = try Self.tapUID(tap)
        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Distil Capture",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapUID,
                    kAudioSubTapDriftCompensationKey: true,
                ]
            ],
        ]
        var aggregate_id = AudioObjectID(kAudioObjectUnknown)
        try Self.check(
            AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &aggregate_id),
            "create aggregate device")
        aggregateID = aggregate_id

        // Delivered on sampleQueue, which keeps the writer fed from one serial
        // context as ConvertingWavWriter requires.
        var proc: AudioDeviceIOProcID?
        try Self.check(
            AudioDeviceCreateIOProcIDWithBlock(&proc, aggregate_id, sampleQueue) {
                [weak self] _, inInputData, _, _, _ in
                self?.append(inInputData)
            }, "install audio IO proc")
        ioProcID = proc

        try Self.check(AudioDeviceStart(aggregate_id, proc), "start audio capture")
        running = true
    }

    func stop() async {
        guard tapID != AudioObjectID(kAudioObjectUnknown) else { return }

        if running, let ioProcID {
            AudioDeviceStop(aggregateID, ioProcID)
        }
        if let ioProcID {
            AudioDeviceDestroyIOProcID(aggregateID, ioProcID)
        }
        if aggregateID != AudioObjectID(kAudioObjectUnknown) {
            AudioHardwareDestroyAggregateDevice(aggregateID)
        }
        AudioHardwareDestroyProcessTap(tapID)

        running = false
        ioProcID = nil
        aggregateID = AudioObjectID(kAudioObjectUnknown)
        tapID = AudioObjectID(kAudioObjectUnknown)

        sampleQueue.sync {}  // drain in-flight callbacks before closing the file
        writer.finalize()
    }

    private func append(_ bufferList: UnsafePointer<AudioBufferList>) {
        guard let sourceFormat,
            let buffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, bufferListNoCopy: bufferList)
        else { return }
        writer.append(buffer)
    }

    // MARK: Core Audio plumbing

    private static func check(_ status: OSStatus, _ what: String) throws {
        guard status != noErr else { return }
        throw CaptureError.message("\(what) failed (OSStatus \(status))")
    }

    /// Our own audio process object, for exclusion from the tap. Empty if the
    /// lookup fails — a tap that records our own output is far better than no
    /// tap at all, and Distil emits no audio during a recording anyway.
    private static func ownProcessObjects() -> [NSNumber] {
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

    private static func defaultOutputDeviceUID() throws -> String {
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
    private static func stringProperty(
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

    private static func tapStreamFormat(_ tap: AudioObjectID) throws
        -> AudioStreamBasicDescription
    {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var asbd = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        try check(
            AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &asbd),
            "read tap format")
        return asbd
    }

    private static func tapUID(_ tap: AudioObjectID) throws -> String {
        try stringProperty(tap, kAudioTapPropertyUID, "read tap UID")
    }
}
