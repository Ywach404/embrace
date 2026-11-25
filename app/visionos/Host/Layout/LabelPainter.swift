import CoreGraphics
import CoreText
import Foundation
import Metal

/// Text and a ring for the layout edit's overlay, painted once on the CPU
/// into small textures in the arm cards' format (sRGB colour
/// premultiplied by alpha, top row first) and kept until the text
/// changes. Compositor thread only.
final class LabelPainter {
    private let device: MTLDevice
    private var cache: [String: MTLTexture] = [:]
    private var order: [String] = []
    private(set) lazy var ring: MTLTexture? = paintRing()

    init(device: MTLDevice) {
        self.device = device
    }

    /// The texture for `text`: light lettering on a dark rounded panel,
    /// one line per line of text, sized to fit.
    func texture(for text: String) -> MTLTexture? {
        if let hit = cache[text] { return hit }
        guard let texture = paint(text) else { return nil }
        cache[text] = texture
        order.append(text)
        if order.count > 24 { cache[order.removeFirst()] = nil }
        return texture
    }

    private func paint(_ text: String) -> MTLTexture? {
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 34, nil)
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map { line -> CTLine in
            let attributes: [NSAttributedString.Key: Any] = [
                NSAttributedString.Key(kCTFontAttributeName as String): font,
                NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(srgbRed: 0.96, green: 0.93, blue: 0.85, alpha: 1),
            ]
            return CTLineCreateWithAttributedString(NSAttributedString(string: String(line), attributes: attributes))
        }
        let lineHeight: CGFloat = 44, pad: CGFloat = 20
        let widest = lines.map { CTLineGetTypographicBounds($0, nil, nil, nil) }.max() ?? 0
        let width = Int(ceil(widest + pad * 2)), height = Int(ceil(CGFloat(lines.count) * lineHeight + pad * 2))
        guard width > 0, height > 0, let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        let panel = CGPath(roundedRect: CGRect(x: 0, y: 0, width: width, height: height), cornerWidth: 18, cornerHeight: 18,
                           transform: nil)
        context.addPath(panel)
        context.setFillColor(CGColor(srgbRed: 0.05, green: 0.03, blue: 0.03, alpha: 0.82))
        context.fillPath()
        context.addPath(panel)
        context.setStrokeColor(CGColor(srgbRed: 0.72, green: 0.55, blue: 0.25, alpha: 0.9))
        context.setLineWidth(3)
        context.strokePath()
        for (i, line) in lines.enumerated() {
            context.textPosition = CGPoint(x: pad, y: CGFloat(height) - pad - CGFloat(i + 1) * lineHeight + 12)
            CTLineDraw(line, context)
        }
        return upload(context, width: width, height: height)
    }

    private func paintRing() -> MTLTexture? {
        let size = 256
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: size * 4,
                                      space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        context.setStrokeColor(CGColor(srgbRed: 1, green: 0.82, blue: 0.4, alpha: 1))
        context.setLineWidth(10)
        context.strokeEllipse(in: CGRect(x: 8, y: 8, width: size - 16, height: size - 16))
        return upload(context, width: size, height: size)
    }

    /// The gizmo's handles, one texture per shape and colour: an arrow
    /// pointing along +x (its quad's width), a thin ring, a square. Kept
    /// apart from the text cache, a handful in all.
    enum Handle: Hashable {
        case arrow, ring, square
    }

    private var handles: [String: MTLTexture] = [:]

    func handle(_ shape: Handle, colour: SIMD3<Float>) -> MTLTexture? {
        let key = "\(shape) \(colour.x) \(colour.y) \(colour.z)"
        if let hit = handles[key] { return hit }
        guard let texture = paintHandle(shape, colour: colour) else { return nil }
        handles[key] = texture
        return texture
    }

    private func paintHandle(_ shape: Handle, colour: SIMD3<Float>) -> MTLTexture? {
        let (width, height): (Int, Int) = shape == .arrow ? (256, 64) : (256, 256)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        let fill = CGColor(srgbRed: CGFloat(colour.x), green: CGFloat(colour.y), blue: CGFloat(colour.z), alpha: 1)
        let edge = CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 0.6)
        switch shape {
        case .arrow:
            let path = CGMutablePath()
            path.move(to: CGPoint(x: 4, y: 26))
            path.addLine(to: CGPoint(x: 196, y: 26))
            path.addLine(to: CGPoint(x: 196, y: 6))
            path.addLine(to: CGPoint(x: 252, y: 32))
            path.addLine(to: CGPoint(x: 196, y: 58))
            path.addLine(to: CGPoint(x: 196, y: 38))
            path.addLine(to: CGPoint(x: 4, y: 38))
            path.closeSubpath()
            context.addPath(path)
            context.setStrokeColor(edge)
            context.setLineWidth(4)
            context.strokePath()
            context.addPath(path)
            context.setFillColor(fill)
            context.fillPath()
        case .ring:
            context.setStrokeColor(edge)
            context.setLineWidth(14)
            context.strokeEllipse(in: CGRect(x: 10, y: 10, width: width - 20, height: height - 20))
            context.setStrokeColor(fill)
            context.setLineWidth(8)
            context.strokeEllipse(in: CGRect(x: 10, y: 10, width: width - 20, height: height - 20))
        case .square:
            let r = CGRect(x: 24, y: 24, width: width - 48, height: height - 48)
            context.setFillColor(fill)
            context.fill(r)
            context.setStrokeColor(edge)
            context.setLineWidth(10)
            context.stroke(r)
        }
        return upload(context, width: width, height: height)
    }

    /// CoreGraphics' rows run bottom up in its coordinates but are stored
    /// top row first, as Metal reads them, so the bytes go across as they
    /// are.
    private func upload(_ context: CGContext, width: Int, height: Int) -> MTLTexture? {
        guard let data = context.data else { return nil }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height,
                                                                  mipmapped: false)
        descriptor.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
        texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: data, bytesPerRow: width * 4)
        return texture
    }
}
