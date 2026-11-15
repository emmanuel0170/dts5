package com.dbe.dts5

/** Maps model-space coordinates back to the source frame. */
class Letterbox(val scale: Float, val padLeft: Int, val padTop: Int, val srcW: Int, val srcH: Int)

/**
 * Resize + RGB + /255 + HWC->CHW in one pass into a preallocated array.
 * Bilinear with OpenCV's half-pixel convention: this is what verify_raw.py validated.
 * No mean/std normalisation: the depth model was verified with plain RGB/255.
 *
 * Input pixels are Android ARGB_8888 ints (Bitmap.getPixels).
 */
object Preprocess {
    private const val PAD_VALUE = 114f / 255f

    /** YOLOE: aspect-preserving resize, centred, grey (114) padding. Matches Ultralytics' rounding. */
    fun letterbox(argb: IntArray, sw: Int, sh: Int, size: Int, out: FloatArray): Letterbox {
        val r = minOf(size.toFloat() / sh, size.toFloat() / sw)
        val nw = Math.round(sw * r)
        val nh = Math.round(sh * r)
        val left = Math.round((size - nw) / 2f - 0.1f)
        val top = Math.round((size - nh) / 2f - 0.1f)
        resizeInto(argb, sw, sh, size, left, top, nw, nh, out)
        return Letterbox(r, left, top, sw, sh)
    }

    /** Depth: plain stretch to size x size (no letterbox), as the released weights were trained. */
    fun stretch(argb: IntArray, sw: Int, sh: Int, size: Int, out: FloatArray) {
        resizeInto(argb, sw, sh, size, 0, 0, size, size, out)
    }

    private fun resizeInto(
        argb: IntArray, sw: Int, sh: Int, size: Int,
        left: Int, top: Int, nw: Int, nh: Int, out: FloatArray,
    ) {
        val plane = size * size
        require(out.size == 3 * plane) { "out must hold 3*$size*$size floats" }
        if (left != 0 || top != 0 || nw != size || nh != size) java.util.Arrays.fill(out, PAD_VALUE)

        val fx = sw.toFloat() / nw
        val fy = sh.toFloat() / nh
        val xa = IntArray(nw)
        val xb = IntArray(nw)
        val xw = FloatArray(nw)
        for (i in 0 until nw) {
            val sx = maxOf((i + 0.5f) * fx - 0.5f, 0f)
            val x0 = minOf(sx.toInt(), sw - 1)
            xa[i] = x0
            xb[i] = minOf(x0 + 1, sw - 1)
            xw[i] = sx - x0
        }
        val inv255 = 1f / 255f
        for (j in 0 until nh) {
            val sy = maxOf((j + 0.5f) * fy - 0.5f, 0f)
            val y0 = minOf(sy.toInt(), sh - 1)
            val y1 = minOf(y0 + 1, sh - 1)
            val wy = sy - y0
            val r0 = y0 * sw
            val r1 = y1 * sw
            val rowOut = (top + j) * size + left
            for (i in 0 until nw) {
                val wx = xw[i]
                val w00 = (1f - wx) * (1f - wy)
                val w01 = wx * (1f - wy)
                val w10 = (1f - wx) * wy
                val w11 = wx * wy
                val p00 = argb[r0 + xa[i]]
                val p01 = argb[r0 + xb[i]]
                val p10 = argb[r1 + xa[i]]
                val p11 = argb[r1 + xb[i]]
                val o = rowOut + i
                out[o] = (((p00 shr 16) and 255) * w00 + ((p01 shr 16) and 255) * w01 +
                    ((p10 shr 16) and 255) * w10 + ((p11 shr 16) and 255) * w11) * inv255
                out[plane + o] = (((p00 shr 8) and 255) * w00 + ((p01 shr 8) and 255) * w01 +
                    ((p10 shr 8) and 255) * w10 + ((p11 shr 8) and 255) * w11) * inv255
                out[2 * plane + o] = ((p00 and 255) * w00 + (p01 and 255) * w01 +
                    (p10 and 255) * w10 + (p11 and 255) * w11) * inv255
            }
        }
    }
}
