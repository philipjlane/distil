import AVFoundation
import CoreMedia
import Foundation

/// Accepts PCM buffers in whatever format the source produces and writes a
/// 16 kHz mono 16-bit WAV (whisper's native input), converting incrementally.
/// Not thread-safe; each instance must be fed from one serial context.
///
/// Optionally mirrors every frame it writes to the WAV into a headerless PCM
/// file for the host's live transcript. A raw file has no length field to go
/// stale, so the host can read it while it grows.
final class ConvertingWavWriter {
    static let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true)!

    private var file: AVAudioFile?
    private var converter: AVAudioConverter?
    private var sourceFormat: AVAudioFormat?
    private(set) var framesWritten: AVAudioFramePosition = 0
    /// Loudest absolute sample written (0…32767). 0 after a whole recording
    /// means the stream was dead — surfaced so silent captures can't hide.
    private(set) var peakAmplitude: Int = 0
    /// The live transcript's copy. Best effort: the WAV is the recording and
    /// this is a preview, so any failure here only switches the copy off.
    private var liveFile: FileHandle?

    init(url: URL, pcmURL: URL? = nil) throws {
        file = try AVAudioFile(
            forWriting: url,
            settings: Self.outputFormat.settings,
            commonFormat: .pcmFormatInt16,
            interleaved: true)
        if let pcmURL {
            if FileManager.default.createFile(atPath: pcmURL.path, contents: nil) {
                liveFile = try? FileHandle(forWritingTo: pcmURL)
            }
            if liveFile == nil {
                // "warning", not "error": the host treats an error event as a
                // failed recording and throws the folder away.
                Emit.event([
                    "event": "warning",
                    "message": "live transcript unavailable: cannot create \(pcmURL.lastPathComponent)",
                ])
            }
        }
    }

    var secondsWritten: Double {
        Double(framesWritten) / Self.outputFormat.sampleRate
    }

    func append(_ sampleBuffer: CMSampleBuffer) {
        guard let pcm = Self.pcmBuffer(from: sampleBuffer) else { return }
        append(pcm)
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        guard buffer.frameLength > 0, file != nil else { return }
        if converter == nil || sourceFormat != buffer.format {
            converter = AVAudioConverter(from: buffer.format, to: Self.outputFormat)
            sourceFormat = buffer.format
        }
        guard let converter else { return }

        let ratio = Self.outputFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: Self.outputFormat, frameCapacity: capacity)
        else { return }

        var fed = false
        var conversionError: NSError?
        let status = converter.convert(to: out, error: &conversionError) { _, inputStatus in
            if fed {
                inputStatus.pointee = .noDataNow
                return nil
            }
            fed = true
            inputStatus.pointee = .haveData
            return buffer
        }
        guard status != .error else {
            Emit.error("audio conversion failed: \(conversionError?.localizedDescription ?? "unknown")")
            return
        }
        write(out)
    }

    /// Flush the resampler tail and close the file (finalises the WAV header).
    func finalize() {
        if let converter, file != nil {
            if let out = AVAudioPCMBuffer(
                pcmFormat: Self.outputFormat,
                frameCapacity: AVAudioFrameCount(Self.outputFormat.sampleRate))
            {
                var conversionError: NSError?
                converter.convert(to: out, error: &conversionError) { _, inputStatus in
                    inputStatus.pointee = .endOfStream
                    return nil
                }
                write(out)
            }
        }
        converter = nil
        file = nil  // AVAudioFile finalises on release
        _ = try? liveFile?.close()
        liveFile = nil
    }

    private func write(_ buffer: AVAudioPCMBuffer) {
        guard buffer.frameLength > 0, let file else { return }
        if let channel = buffer.int16ChannelData?.pointee {
            for i in 0..<Int(buffer.frameLength) {
                let magnitude = Int(channel[i].magnitude)
                if magnitude > peakAmplitude { peakAmplitude = magnitude }
            }
        }
        do {
            try file.write(from: buffer)
            framesWritten += AVAudioFramePosition(buffer.frameLength)
            writePCM(buffer)
        } catch {
            Emit.error("wav write failed: \(error.localizedDescription)")
        }
    }

    /// Append the frames just written to the WAV to the live transcript's file.
    /// `write(contentsOf:)` throws; the older `write(_:)` raises an Objective-C
    /// exception on a full disk, which would take the recording down with it.
    private func writePCM(_ buffer: AVAudioPCMBuffer) {
        guard let liveFile, let channel = buffer.int16ChannelData?.pointee else { return }
        do {
            try liveFile.write(
                contentsOf: Data(bytes: channel, count: Int(buffer.frameLength) * MemoryLayout<Int16>.size))
        } catch {
            // Say so once and stop; the WAV is untouched. "warning" for the
            // same reason as in init.
            Emit.event([
                "event": "warning",
                "message": "live transcript stopped: \(error.localizedDescription)",
            ])
            _ = try? liveFile.close()
            self.liveFile = nil
        }
    }

    private static func pcmBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let description = CMSampleBufferGetFormatDescription(sampleBuffer),
            let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description)
        else { return nil }
        let frames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard frames > 0,
            let format = AVAudioFormat(streamDescription: asbd),
            let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)
        else { return nil }
        pcm.frameLength = frames
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer, at: 0, frameCount: Int32(frames), into: pcm.mutableAudioBufferList)
        return status == noErr ? pcm : nil
    }
}
