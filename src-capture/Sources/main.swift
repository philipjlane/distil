import Foundation

let arguments = CommandLine.arguments

if arguments.contains("--check-permissions") {
    PermissionCheck.runAndPrint()
    exit(0)
}

guard let flagIndex = arguments.firstIndex(of: "--out-dir"), flagIndex + 1 < arguments.count else {
    FileHandle.standardError.write(Data("usage: distil-capture --out-dir <dir> | --check-permissions\n".utf8))
    exit(2)
}

do {
    let outDir = URL(fileURLWithPath: arguments[flagIndex + 1], isDirectory: true)
    let session = try CaptureSession(outDir: outDir)
    session.start()
    StopTriggers.install { session.requestStop() }
} catch {
    Emit.error("failed to initialise capture: \(error.localizedDescription)")
    exit(1)
}

// Park the main thread and service the main dispatch queue (signal sources,
// MainActor tasks). The session calls exit() when it finishes.
dispatchMain()
