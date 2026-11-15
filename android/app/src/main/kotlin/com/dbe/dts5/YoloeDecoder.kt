package com.dbe.dts5

class Detection(
    val classId: Int,
    val score: Float,
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    /** Anchor index in the 8400-anchor head; needed later to decode this detection's mask. */
    val anchor: Int,
)

/**
 * Decodes output 0 of the exported YOLOE model, as verified by verify_raw.py:
 *  - flat layout of tensor [1, 4 + nc + 32, 8400], i.e. value(channel c, anchor a) = out[c * 8400 + a]
 *  - channels 0..3   : box centre-x, centre-y, width, height, NORMALISED to 0..1 (multiply by 640)
 *  - channels 4..4+nc: class scores, already probabilities (sigmoid applied in the graph)
 *  - last 32 channels: mask coefficients (unused here)
 *  - NMS: class-agnostic, conf 0.25, IoU 0.7 (Ultralytics' YOLOE defaults)
 */
class YoloeDecoder(
    private val numClasses: Int,
    private val inputSize: Int = 640,
    /** 32 for the seg export, 0 for the detection-only exports. */
    private val maskCoefs: Int = 32,
) {
    companion object {
        const val ANCHORS = 8400
        const val MASK_COEFS = 32
    }

    val expectedSize: Int get() = (4 + numClasses + maskCoefs) * ANCHORS

    private val bestScore = FloatArray(ANCHORS)
    private val bestClass = IntArray(ANCHORS)

    fun decode(
        out: FloatArray, lb: Letterbox,
        confThr: Float = 0.25f, iouThr: Float = 0.7f, maxDet: Int = 100,
    ): List<Detection> {
        require(out.size == expectedSize) { "output size ${out.size} != expected $expectedSize" }

        // Class-outer / anchor-inner loop keeps memory access sequential.
        java.util.Arrays.fill(bestScore, 0f)
        for (c in 0 until numClasses) {
            val base = (4 + c) * ANCHORS
            for (a in 0 until ANCHORS) {
                val s = out[base + a]
                if (s > bestScore[a]) {
                    bestScore[a] = s
                    bestClass[a] = c
                }
            }
        }

        val candidates = (0 until ANCHORS).filter { bestScore[it] > confThr }.sortedByDescending { bestScore[it] }
        val sz = inputSize.toFloat()
        val kept = ArrayList<Detection>() // model space (640x640 letterboxed)
        for (a in candidates) {
            if (kept.size >= maxDet) break
            val cx = out[a] * sz
            val cy = out[ANCHORS + a] * sz
            val w = out[2 * ANCHORS + a] * sz
            val h = out[3 * ANCHORS + a] * sz
            val x1 = cx - w / 2
            val y1 = cy - h / 2
            val x2 = cx + w / 2
            val y2 = cy + h / 2
            var suppressed = false
            for (k in kept) {
                if (overlap(k.x1, k.y1, k.x2, k.y2, x1, y1, x2, y2) > iouThr) {
                    suppressed = true
                    break
                }
            }
            if (!suppressed) kept.add(Detection(bestClass[a], bestScore[a], x1, y1, x2, y2, a))
        }
        return kept.map { toSource(it, lb) }
    }

    private fun toSource(d: Detection, lb: Letterbox): Detection {
        fun mx(v: Float) = ((v - lb.padLeft) / lb.scale).coerceIn(0f, lb.srcW.toFloat())
        fun my(v: Float) = ((v - lb.padTop) / lb.scale).coerceIn(0f, lb.srcH.toFloat())
        return Detection(d.classId, d.score, mx(d.x1), my(d.y1), mx(d.x2), my(d.y2), d.anchor)
    }

    private fun overlap(
        ax1: Float, ay1: Float, ax2: Float, ay2: Float,
        bx1: Float, by1: Float, bx2: Float, by2: Float,
    ): Float {
        val iw = maxOf(0f, minOf(ax2, bx2) - maxOf(ax1, bx1))
        val ih = maxOf(0f, minOf(ay2, by2) - maxOf(ay1, by1))
        val inter = iw * ih
        val union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return if (union > 0f) inter / union else 0f
    }
}
