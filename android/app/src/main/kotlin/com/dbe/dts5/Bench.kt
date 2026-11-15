package com.dbe.dts5

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.BitmapFactory
import android.os.BatteryManager
import android.os.Build
import android.util.Log
import com.google.ai.edge.litert.Environment
import java.io.File
import java.util.Locale
import kotlin.math.abs

/**
 * Benchmark v2. Differences from v1:
 *  - GPU_DEFAULT only (CPU is unusable, FP32 gave no accuracy benefit at 2x the latency).
 *  - Reports END-TO-END latency (run + read + decode). v1's "write+run" excluded GPU compute because
 *    run() appears to only enqueue work; the wait happens inside read().
 *  - "probe": run, sleep, then time read(). If read() collapses after the pause, the earlier read time was
 *    waiting on the GPU, not copying.
 *  - Model variants (edit the lists below to match the files you put in assets/).
 *  - Two sustained runs (heaviest and lightest pair) with both models loaded together, logging cycle time and battery temperature.
 * Let the phone cool down before running; thermal state affects every number here.
 */
class Bench(private val ctx: Context) {
    class YoloeVariant(val label: String, val asset: String, val maskCoefs: Int)
    class DepthVariant(val label: String, val asset: String, val size: Int)

    // A variant whose asset is missing just logs FAILED.
    private val yoloeVariants = listOf(
        YoloeVariant("26s-det", "yoloe-26s-det.tflite", 0),
        YoloeVariant("26n-det", "yoloe-26n-det.tflite", 0),
    )
    private val depthVariants = listOf(
        DepthVariant("s-768", "yolo26s-depth-768.tflite", 768),
        DepthVariant("s-640", "yolo26s-depth-640.tflite", 640),
        DepthVariant("n-768", "yolo26n-depth-768.tflite", 768),
    )
    private val backend = Backend.GPU_DEFAULT

    private val sb = StringBuilder()
    private val file = File(ctx.filesDir, "bench.txt")

    private fun log(s: String) {
        Log.i(TAG, s)
        sb.appendLine(s)
        runCatching { file.writeText(sb.toString()) }
    }

    fun run(): String {
        sb.clear()
        val bmp = ctx.assets.open("bus.jpg").use { BitmapFactory.decodeStream(it) }
        val w = bmp.width
        val h = bmp.height
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        val labels = ctx.assets.open("labels.txt").bufferedReader().readLines().filter { it.isNotBlank() }

        val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else ""
        log("device: ${Build.MANUFACTURER} ${Build.MODEL} hw=${Build.HARDWARE} soc=$soc sdk=${Build.VERSION.SDK_INT}")
        log("battery temp at start: %.1f C".fmt(batteryC()))
        log("image ${w}x$h, labels=${labels.size}, backend=${backend.name}, iterations=$ITER")

        val env = Environment.create()
        log("")
        for (v in yoloeVariants) benchYoloe(v, env, px, w, h, labels)
        log("")
        for (v in depthVariants) benchDepth(v, env, px, w, h)
        log("")
        sustained(env, px, w, h, labels, yoloeVariants[0], depthVariants[0]) // heaviest: 26s-det + s-768
        log("")
        sustained(env, px, w, h, labels, yoloeVariants[1], depthVariants[1]) // lightest: 26n-det + s-640
        log("")
        log("done")
        return sb.toString()
    }

    private fun benchYoloe(v: YoloeVariant, env: Environment, px: IntArray, w: Int, h: Int, labels: List<String>) {
        val tag = "YOLOE ${v.label}"
        try {
            val input = FloatArray(3 * 640 * 640)
            val lb = Preprocess.letterbox(px, w, h, 640, input)
            val dec = YoloeDecoder(labels.size, maskCoefs = v.maskCoefs)
            val t0 = System.nanoTime()
            ModelRunner(ctx, v.asset, backend, env).use { m ->
                val compileMs = (System.nanoTime() - t0) / 1e6
                m.run(input)
                var idx = -1
                for (i in 0 until m.outputCount) if (m.read(i).size == dec.expectedSize) idx = i
                check(idx >= 0) { "no output of size ${dec.expectedSize} (outputs: ${m.outputCount})" }
                repeat(WARMUP) { m.run(input); m.read(idx) }

                val run = DoubleArray(ITER)
                val read = DoubleArray(ITER)
                val post = DoubleArray(ITER)
                val total = DoubleArray(ITER)
                var dets = emptyList<Detection>()
                for (i in 0 until ITER) {
                    val t1 = System.nanoTime()
                    m.run(input)
                    val t2 = System.nanoTime()
                    val out = m.read(idx)
                    val t3 = System.nanoTime()
                    dets = dec.decode(out, lb)
                    val t4 = System.nanoTime()
                    run[i] = (t2 - t1) / 1e6
                    read[i] = (t3 - t2) / 1e6
                    post[i] = (t4 - t3) / 1e6
                    total[i] = (t4 - t1) / 1e6
                }
                val probe = probe(m, input, idx)
                log("$tag: END-TO-END ${stat(total)}   (compile %.0f ms)".fmt(compileMs))
                log("   run ${stat(run)} | read ${stat(read)} | decode ${stat(post)}")
                log("   probe: read after ${PROBE_SLEEP_MS} ms pause ${stat(probe)}")
                log("   detections=${dets.size}; top5: " +
                    dets.take(5).joinToString("; ") { "${labels[it.classId]} %.2f".fmt(it.score) })
            }
        } catch (t: Throwable) {
            log("$tag: FAILED: ${t.javaClass.simpleName}: ${t.message}")
        }
    }

    private fun benchDepth(v: DepthVariant, env: Environment, px: IntArray, w: Int, h: Int) {
        val tag = "DEPTH ${v.label}"
        try {
            val input = FloatArray(3 * v.size * v.size)
            Preprocess.stretch(px, w, h, v.size, input)
            val t0 = System.nanoTime()
            ModelRunner(ctx, v.asset, backend, env).use { m ->
                val compileMs = (System.nanoTime() - t0) / 1e6
                repeat(WARMUP) { m.run(input); m.read(0) }
                val run = DoubleArray(ITER)
                val read = DoubleArray(ITER)
                val total = DoubleArray(ITER)
                var out = FloatArray(0)
                for (i in 0 until ITER) {
                    val t1 = System.nanoTime()
                    m.run(input)
                    val t2 = System.nanoTime()
                    out = m.read(0)
                    val t3 = System.nanoTime()
                    run[i] = (t2 - t1) / 1e6
                    read[i] = (t3 - t2) / 1e6
                    total[i] = (t3 - t1) / 1e6
                }
                check(out.size == v.size * v.size) { "depth output size ${out.size} != ${v.size * v.size}" }
                var mn = Float.MAX_VALUE
                var mx = -Float.MAX_VALUE
                var sum = 0.0
                var bad = 0
                for (x in out) {
                    if (x.isNaN() || x.isInfinite()) { bad++; continue }
                    if (x < mn) mn = x
                    if (x > mx) mx = x
                    sum += x
                }
                val mean = (sum / out.size).toFloat()
                val probe = probe(m, input, 0)
                log("$tag: END-TO-END ${stat(total)}   (compile %.0f ms)".fmt(compileMs))
                log("   run ${stat(run)} | read ${stat(read)}")
                log("   probe: read after ${PROBE_SLEEP_MS} ms pause ${stat(probe)}")
                log("   metres min/mean/max = %.2f/%.2f/%.2f; mean vs python s-768 (%.2f): %+.1f%%; nan/inf=%d"
                    .fmt(mn, mean, mx, REF_DEPTH_MEAN, 100f * (mean - REF_DEPTH_MEAN) / REF_DEPTH_MEAN, bad))
            }
        } catch (t: Throwable) {
            log("$tag: FAILED: ${t.javaClass.simpleName}: ${t.message}")
        }
    }

    /** run(), pause, then time read(). Small time => the earlier read() was waiting for the GPU. */
    private fun probe(m: ModelRunner, input: FloatArray, idx: Int): DoubleArray {
        val out = DoubleArray(PROBE_N)
        for (i in 0 until PROBE_N) {
            m.run(input)
            Thread.sleep(PROBE_SLEEP_MS)
            val t = System.nanoTime()
            m.read(idx)
            out[i] = (System.nanoTime() - t) / 1e6
        }
        return out
    }

    /** Two models loaded together, run back-to-back for SUSTAINED_S seconds. */
    private fun sustained(
        env: Environment, px: IntArray, w: Int, h: Int, labels: List<String>,
        yv: YoloeVariant, dv: DepthVariant,
    ) {
        log("SUSTAINED ${SUSTAINED_S}s: ${yv.label} + ${dv.label}, sequential per cycle (preprocess included)")
        try {
            val yIn = FloatArray(3 * 640 * 640)
            val dIn = FloatArray(3 * dv.size * dv.size)
            val lb = Preprocess.letterbox(px, w, h, 640, yIn)
            val dec = YoloeDecoder(labels.size, maskCoefs = yv.maskCoefs)
            ModelRunner(ctx, yv.asset, backend, env).use { ym ->
                ModelRunner(ctx, dv.asset, backend, env).use { dm ->
                    ym.run(yIn)
                    var yIdx = -1
                    for (i in 0 until ym.outputCount) if (ym.read(i).size == dec.expectedSize) yIdx = i
                    check(yIdx >= 0) { "no YOLOE output of size ${dec.expectedSize}" }

                    val start = System.nanoTime()
                    var windowStart = start
                    val cycles = ArrayList<Double>()
                    while ((System.nanoTime() - start) / 1e9 < SUSTAINED_S) {
                        val t0 = System.nanoTime()
                        Preprocess.letterbox(px, w, h, 640, yIn)
                        ym.run(yIn)
                        dec.decode(ym.read(yIdx), lb)
                        Preprocess.stretch(px, w, h, dv.size, dIn)
                        dm.run(dIn)
                        dm.read(0)
                        cycles.add((System.nanoTime() - t0) / 1e6)
                        if ((System.nanoTime() - windowStart) / 1e9 >= 10) {
                            val med = cycles.sorted()[cycles.size / 2]
                            log("   t=%3.0fs  cycle med %.0f ms (%.2f cycles/s)  battery %.1f C"
                                .fmt((System.nanoTime() - start) / 1e9, med, 1000.0 / med, batteryC()))
                            cycles.clear()
                            windowStart = System.nanoTime()
                        }
                    }
                }
            }
        } catch (t: Throwable) {
            log("SUSTAINED FAILED: ${t.javaClass.simpleName}: ${t.message}")
        }
    }

    private fun batteryC(): Float {
        val i = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return (i?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10f
    }

    private fun String.fmt(vararg a: Any) = String.format(Locale.US, this, *a)

    private fun stat(ms: DoubleArray): String {
        val s = ms.sortedArray()
        return String.format(Locale.US, "med %.1f / min %.1f ms", s[s.size / 2], s.first())
    }

    companion object {
        private const val TAG = "SpatialBench"
        private const val ITER = 15
        private const val WARMUP = 3
        private const val PROBE_N = 5
        private const val PROBE_SLEEP_MS = 400L
        private const val SUSTAINED_S = 60
        private const val REF_DEPTH_MEAN = 4.4842f // python, depth-s 768, bus.jpg

        fun lastReport(ctx: Context): String {
            val f = File(ctx.filesDir, "bench.txt")
            return if (f.exists()) f.readText() else ""
        }
    }
}
