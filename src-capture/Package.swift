// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "distil-capture",
    // 14.2 is the floor for Core Audio process taps (AudioHardwareCreateProcessTap),
    // which is how system audio is captured without the Screen Recording permission.
    platforms: [.macOS("14.2")],
    targets: [
        .executableTarget(
            name: "distil-capture",
            path: "Sources",
            linkerSettings: [
                // Embed an Info.plist so the bare CLI has a bundle identity and
                // mic usage description when run outside the app bundle.
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Resources/Info.plist",
                ])
            ]
        )
    ]
)
