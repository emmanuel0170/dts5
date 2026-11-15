package com.dbe.dts5

class DepthStat(val median: Float, val p10: Float)

/**
 * Samples a depth map (metres, size x size, stretched from the upright frame, so normalised frame
 * coordinates map straight to depth pixels).
 * First-version approximation: per-object depth comes from the central part of the BOX, not a mask.
 */
object DepthSampler {
    /** Median and 10th-percentile (nearer-biased) depth over a sample grid on the central part of a box (0..1 coords). */
    fun boxStats(
        depth: FloatArray, size: Int,
        x1: Float, y1: Float, x2: Float, y2: Float,
        scratch: FloatArray, core: Float = 0.5f, n: Int = 12,
    ): DepthStat {
        val cx = (x1 + x2) / 2
        val cy = (y1 + y2) / 2
        val hw = (x2 - x1) * core / 2
        val hh = (y2 - y1) * core / 2
        var k = 0
        for (j in 0 until n) {
            val py = ((cy - hh + 2 * hh * (j + 0.5f) / n) * size).toInt().coerceIn(0, size - 1)
            for (i in 0 until n) {
                val px = ((cx - hw + 2 * hw * (i + 0.5f) / n) * size).toInt().coerceIn(0, size - 1)
                val v = depth[py * size + px]
                if (v.isFinite()) scratch[k++] = v
            }
        }
        if (k == 0) return DepthStat(Float.NaN, Float.NaN)
        java.util.Arrays.sort(scratch, 0, k)
        return DepthStat(scratch[k / 2], scratch[k / 10])
    }

    /** rows x cols grid over the whole frame; each cell is its 10th-percentile depth (nearer-biased), row-major. */
    fun gridStats(depth: FloatArray, size: Int, rows: Int, cols: Int, scratch: FloatArray, n: Int = 6): List<Double> {
        val out = ArrayList<Double>(rows * cols)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                var k = 0
                for (j in 0 until n) {
                    val py = (((r + (j + 0.5f) / n) / rows) * size).toInt().coerceIn(0, size - 1)
                    for (i in 0 until n) {
                        val px = (((c + (i + 0.5f) / n) / cols) * size).toInt().coerceIn(0, size - 1)
                        val v = depth[py * size + px]
                        if (v.isFinite()) scratch[k++] = v
                    }
                }
                if (k == 0) out.add(Double.NaN) else {
                    java.util.Arrays.sort(scratch, 0, k)
                    out.add(scratch[k / 10].toDouble())
                }
            }
        }
        return out
    }

    /**
     * Downsamples the size x size depth map (stretched from the upright frame) to gw x gh: the median of a 3x3 sample
     * pattern per cell, NaN when fewer than 5 of the 9 samples are valid. Row 0 is the top of the upright frame.
     * This is the grid the Dart path planner works on.
     */
    fun downsample(depth: FloatArray, size: Int, gw: Int, gh: Int, out: FloatArray, scratch: FloatArray) {
        for (j in 0 until gh) {
            for (i in 0 until gw) {
                var k = 0
                for (sj in 0 until 3) {
                    val py = (((j + (sj + 0.5f) / 3f) / gh) * size).toInt().coerceIn(0, size - 1)
                    for (si in 0 until 3) {
                        val px = (((i + (si + 0.5f) / 3f) / gw) * size).toInt().coerceIn(0, size - 1)
                        val v = depth[py * size + px]
                        if (v.isFinite() && v > 0f) scratch[k++] = v
                    }
                }
                if (k < 5) {
                    out[j * gw + i] = Float.NaN
                } else {
                    java.util.Arrays.sort(scratch, 0, k)
                    out[j * gw + i] = scratch[k / 2]
                }
            }
        }
    }
}
