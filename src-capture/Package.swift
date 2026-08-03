// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "distil-capture",
    platforms: [.macOS(.v14)],
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
