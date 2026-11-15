package com.dbe.dts5

/** Everything the engine needs to know; sent from Dart so variants can be tried without rebuilding. */
class SceneConfig(
    val yoloeAsset: String,
    val yoloeMaskCoefs: Int,
    val depthAsset: String,
    val depthSize: Int,
    /** Depth runs on every Nth frame; detections run on every frame. 1 = both every frame. */
    val depthEvery: Int,
    val confThr: Float,
    val iouThr: Float,
    /** Enqueue the depth model before reading YOLOE's output so CPU work overlaps GPU work. */
    val overlap: Boolean,
) {
    companion object {
        fun fromMap(m: Map<*, *>) = SceneConfig(
            yoloeAsset = m["yoloeAsset"] as String,
            yoloeMaskCoefs = (m["yoloeMaskCoefs"] as Number).toInt(),
            depthAsset = m["depthAsset"] as String,
            depthSize = (m["depthSize"] as Number).toInt(),
            depthEvery = maxOf(1, (m["depthEvery"] as Number).toInt()),
            confThr = (m["confThr"] as Number).toFloat(),
            iouThr = (m["iouThr"] as Number).toFloat(),
            overlap = m["overlap"] as Boolean,
        )
    }
}
