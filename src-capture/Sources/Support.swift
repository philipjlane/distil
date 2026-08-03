import Foundation

enum Emit {
    private static let queue = DispatchQueue(label: "emit")

    static func event(_ payload: [String: Any]) {
        queue.sync {
            guard let data = try? JSONSerialization.data(withJSONObject: payload),
                let line = String(data: data, encoding: .utf8)
            else { return }
            print(line)
            fflush(stdout)
        }
    }

    static func error(_ message: String) {
        event(["event": "error", "message": message])
    }
}

enum StopTriggers {
    private static var signalSources: [DispatchSourceSignal] = []

    /// Stop on: a "stop" line from the host app, stdin EOF (host died — never
    /// keep recording as an orphan), SIGINT, or SIGTERM.
    static func install(onStop: @escaping () -> Void) {
        DispatchQueue.global().async {
            while let line = readLine(strippingNewline: true) {
                if line == "stop" { break }
            }
            onStop()
        }
        for sig in [SIGINT, SIGTERM] {
            signal(sig, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            source.setEventHandler { onStop() }
            source.resume()
            signalSources.append(source)
        }
    }
}
