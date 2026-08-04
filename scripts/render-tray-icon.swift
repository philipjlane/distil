// Renders the menu-bar template icon from an SF Symbol (funnel — distil).
// Usage: swift scripts/render-tray-icon.swift src-tauri/icons/tray.png
import AppKit

let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "tray.png"
let candidates = [
    "funnel", "line.3.horizontal.decrease.circle", "drop.fill", "drop",
]

var symbol: NSImage?
for name in candidates {
    if let image = NSImage(systemSymbolName: name, accessibilityDescription: nil) {
        symbol = image.withSymbolConfiguration(
            NSImage.SymbolConfiguration(pointSize: 32, weight: .medium))
        print("using SF Symbol: \(name)")
        break
    }
}
guard let symbol else { fatalError("no usable SF Symbol found") }

let canvas = NSSize(width: 44, height: 44)
let rendered = NSImage(size: canvas)
rendered.lockFocus()
let symbolSize = symbol.size
let scale = min(36 / symbolSize.width, 36 / symbolSize.height)
let drawSize = NSSize(width: symbolSize.width * scale, height: symbolSize.height * scale)
let origin = NSPoint(
    x: (canvas.width - drawSize.width) / 2, y: (canvas.height - drawSize.height) / 2)
NSColor.black.set()
symbol.draw(
    in: NSRect(origin: origin, size: drawSize),
    from: .zero,
    operation: .sourceOver,
    fraction: 1.0)
rendered.unlockFocus()

guard let tiff = rendered.tiffRepresentation,
    let rep = NSBitmapImageRep(data: tiff),
    let png = rep.representation(using: .png, properties: [:])
else { fatalError("png encode failed") }
try! png.write(to: URL(fileURLWithPath: out))
print("wrote \(out)")
