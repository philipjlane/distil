import Foundation

let arguments = CommandLine.arguments

if arguments.contains("--check-permissions") {
    PermissionCheck.runAndPrint()
    exit(0)
}

if arguments.contains("--check-system-audio") {
    SystemAudioPermission.probeAndPrint()
    exit(0)
}

if let index = arguments.firstIndex(of: "--request-permission"), index + 1 < arguments.count {
    let kind = arguments[index + 1]
    Task {
        await PermissionRequest.runAndPrint(kind: kind)
        exit(0)
    }
    dispatchMain()
}

if arguments.contains("calendar") {
    var hours = 12.0
    if let index = arguments.firstIndex(of: "--hours"), index + 1 < arguments.count,
        let parsed = Double(arguments[index + 1])
    {
        hours = parsed
    }
    Task { await CalendarReporter.runAndPrint(hours: hours) }
    dispatchMain()
}

if arguments.contains("watch-mic") {
    MicActivityWatcher.run()
    dispatchMain()
}

guard let flagIndex = arguments.firstIndex(of: "--out-dir"), flagIndex + 1 < arguments.count else {
    FileHandle.standardError.write(
        Data(
            "usage: distil-capture --out-dir <dir> [--live-pcm] | --check-permissions | --check-system-audio | --request-permission <microphone|calendar> | watch-mic\n"
                .utf8))
    exit(2)
}

do {
    let outDir = URL(fileURLWithPath: arguments[flagIndex + 1], isDirectory: true)
    let session = try CaptureSession(outDir: outDir, livePCM: arguments.contains("--live-pcm"))
    session.start()
    StopTriggers.install { session.requestStop() }
} catch {
    Emit.error("failed to initialise capture: \(error.localizedDescription)")
    exit(1)
}

// Park the main thread and service the main dispatch queue (signal sources,
// MainActor tasks). The session calls exit() when it finishes.
dispatchMain()
