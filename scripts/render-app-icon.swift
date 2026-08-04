// Renders the 1024x1024 app-icon source: distil glyph on a dark rounded tile.
// Usage: swift scripts/render-app-icon.swift /tmp/appicon.png
// Then:  npm run tauri icon /tmp/appicon.png
import AppKit

let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "appicon.png"
let side: CGFloat = 1024

let candidates = ["funnel", "line.3.horizontal.decrease.circle", "drop.fill"]
var symbol: NSImage?
for name in candidates {
    if let image = NSImage(systemSymbolName: name, accessibilityDescription: nil) {
        symbol = image.withSymbolConfiguration(
            NSImage.SymbolConfiguration(pointSize: 600, weight: .medium))
        print("using SF Symbol: \(name)")
        break
    }
}
guard let symbol else { fatalError("no usable SF Symbol found") }

let canvas = NSSize(width: side, height: side)
let rendered = NSImage(size: canvas)
rendered.lockFocus()

// macOS icon grid: content tile is ~824pt of the 1024 canvas, corner radius ~185.
let inset: CGFloat = 100
let tile = NSRect(x: inset, y: inset, width: side - inset * 2, height: side - inset * 2)
let path = NSBezierPath(roundedRect: tile, xRadius: 185, yRadius: 185)
let gradient = NSGradient(
    starting: NSColor(calibratedRed: 0.16, green: 0.17, blue: 0.20, alpha: 1),
    ending: NSColor(calibratedRed: 0.09, green: 0.10, blue: 0.12, alpha: 1))
gradient?.draw(in: path, angle: -90)

// Glyph, white, centred, ~55% of the tile.
let symbolSize = symbol.size
let target = tile.width * 0.55
let scale = min(target / symbolSize.width, target / symbolSize.height)
let drawSize = NSSize(width: symbolSize.width * scale, height: symbolSize.height * scale)
let origin = NSPoint(
    x: (side - drawSize.width) / 2, y: (side - drawSize.height) / 2)
let tinted = NSImage(size: symbolSize)
tinted.lockFocus()
NSColor.white.set()
let symbolRect = NSRect(origin: .zero, size: symbolSize)
symbol.draw(in: symbolRect)
symbolRect.fill(using: .sourceAtop)
tinted.unlockFocus()
tinted.draw(
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
