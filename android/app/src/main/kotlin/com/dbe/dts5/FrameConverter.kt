package com.dbe.dts5

import androidx.camera.core.ImageProxy

/** An upright (rotation applied) frame as ARGB ints, ready for Preprocess. */
class UprightFrame {
    var w = 0
    var h = 0
    var argb = IntArray(0)
}

/**
 * Converts CameraX RGBA_8888 frames (bytes in R,G,B,A order, possibly padded rows) to upright ARGB ints,
 * applying imageInfo.rotationDegrees (clockwise degrees needed to make the image upright).
 * Reuses its buffers between frames.
 */
class FrameConverter {
    private var bytes = ByteArray(0)
    val frame = UprightFrame()

    fun convert(image: ImageProxy) {
        val plane = image.planes[0]
        val buf = plane.buffer
        buf.rewind()
        val n = buf.remaining()
        if (bytes.size < n) bytes = ByteArray(n)
        buf.get(bytes, 0, n)

        val sw = image.width
        val sh = image.height
        val rs = plane.rowStride
        val rot = image.imageInfo.rotationDegrees
        val dw = if (rot == 90 || rot == 270) sh else sw
        val dh = if (rot == 90 || rot == 270) sw else sh
        if (frame.argb.size != dw * dh) frame.argb = IntArray(dw * dh)
        frame.w = dw
        frame.h = dh
        val out = frame.argb

        // destination index = base + x * xs + y * ys, for each rotation
        val base: Int
        val xs: Int
        val ys: Int
        when (rot) {
            90 -> { base = sh - 1; xs = dw; ys = -1 }
            180 -> { base = (sh - 1) * dw + (sw - 1); xs = -1; ys = -dw }
            270 -> { base = (sw - 1) * dw; xs = -dw; ys = 1 }
            else -> { base = 0; xs = 1; ys = dw }
        }
        for (y in 0 until sh) {
            var o = y * rs
            var d = base + y * ys
            for (x in 0 until sw) {
                val r = bytes[o].toInt() and 255
                val g = bytes[o + 1].toInt() and 255
                val b = bytes[o + 2].toInt() and 255
                out[d] = (255 shl 24) or (r shl 16) or (g shl 8) or b
                o += 4
                d += xs
            }
        }
    }
}
