package com.dbe.dts5

import android.content.Context
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Handler
import android.os.Looper
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.ai.edge.litert.Environment
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Headless scene engine: CameraX ImageAnalysis only (no Preview use case is bound, so there is no camera view),
 * YOLOE detection + metric depth on the GPU, fused into one event per camera frame.
 *
 * Threading: models are created AND used only on one single-thread executor (GPU contexts are thread-affine).
 * start()/stop() must be called from the main thread. Frames are delivered to onFrame on the main thread.
 *
 * Fusion (first version): per-detection depth = median / 10th percentile over the central half of the box,
 * read from the latest depth map. When depthEvery > 1 that map can be several frames old: depthAgeMs says how old.
 */
class SceneEngine(
    private val ctx: Context,
    private val owner: LifecycleOwner,
    private val onFrame: (HashMap<String, Any?>) -> Unit,
    private val onError: (String) -> Unit,
) {
    private val main = Handler(Looper.getMainLooper())
    @Volatile private var exec: ExecutorService? = null
    private var provider: ProcessCameraProvider? = null

    // ---- created and used only on the `exec` thread ----
    private var cfg: SceneConfig? = null
    private var env: Environment? = null
    private var labels: List<String> = emptyList()
    private var decoder: YoloeDecoder? = null
    private var yoloe: ModelRunner? = null
    private var depth: ModelRunner? = null
    private var yIdx = -1
    private var yIn = FloatArray(0)
    private var dIn = FloatArray(0)
    private val conv = FrameConverter()
    private var latestDepth: FloatArray? = null
    private var latestDepthNs = 0L
    private var frameId = 0L
    private val scratch = FloatArray(512)
    @Volatile private var viz = false // stream a camera preview for the visualisation screens
    @Volatile private var hfovDeg = 60f // horizontal field of view of the upright frame; read from the camera if possible
    private val gridBuf = FloatArray(GRID_W * GRID_H)
    private val previewPx = IntArray(PREVIEW_W * PREVIEW_H)

    fun start(config: SceneConfig) {
        stop()
        val ex = Executors.newSingleThreadExecutor()
        exec = ex
        ex.execute {
            try {
                loadModels(config)
                main.post { bindCamera(ex) }
            } catch (t: Throwable) {
                fail(t)
            }
        }
    }

    /** Visualisation screens switch this on while they are visible: it adds a small JPEG preview to every event. */
    fun setViz(on: Boolean) {
        viz = on
    }

    /**
     * Horizontal field of view of the upright (portrait) frame from the back camera's sensor size and focal length.
     * The upright horizontal axis is the sensor's short side. Approximate: assumes the 4:3 analysis stream uses the
     * full sensor width and that the first back camera is the one CameraX picked.
     */
    private fun readHfov(): Float? {
        return try {
            val cm = ctx.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val id = cm.cameraIdList.firstOrNull {
                cm.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            } ?: return null
            val ch = cm.getCameraCharacteristics(id)
            val size = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return null
            val focal = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: return null
            val shortSide = minOf(size.width, size.height)
            val deg = Math.toDegrees(2.0 * Math.atan(shortSide / (2.0 * focal))).toFloat()
            if (deg in 30f..110f) deg else null
        } catch (t: Throwable) {
            null
        }
    }

    /** Downscaled upright frame as a JPEG for the visualisation screens (engine thread). */
    private fun previewJpeg(f: UprightFrame): ByteArray {
        val sx = f.w.toFloat() / PREVIEW_W
        val sy = f.h.toFloat() / PREVIEW_H
        for (y in 0 until PREVIEW_H) {
            val row = minOf((y * sy).toInt(), f.h - 1) * f.w
            for (x in 0 until PREVIEW_W) previewPx[y * PREVIEW_W + x] = f.argb[row + minOf((x * sx).toInt(), f.w - 1)]
        }
        val bmp = Bitmap.createBitmap(previewPx, PREVIEW_W, PREVIEW_H, Bitmap.Config.ARGB_8888)
        val bos = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, 65, bos)
        bmp.recycle()
        return bos.toByteArray()
    }

    fun stop() {
        runCatching { provider?.unbindAll() } // unbind first so no new frames reach a shut-down executor
        val ex = exec ?: return
        exec = null
        ex.execute { closeModels() }
        ex.shutdown()
    }

    /**
     * JPEG of the most recent upright camera frame (null if there is none yet or the engine is stopped).
     * Built on the engine thread, which owns the frame buffer; the callback runs on the main thread.
     */
    fun snapshot(quality: Int = 80, onJpeg: (ByteArray?) -> Unit) {
        val ex = exec
        if (ex == null) {
            onJpeg(null)
            return
        }
        try {
            ex.execute {
                val f = conv.frame
                val jpeg: ByteArray? = try {
                    if (f.w == 0 || f.h == 0) {
                        null
                    } else {
                        val bmp = Bitmap.createBitmap(f.argb, f.w, f.h, Bitmap.Config.ARGB_8888)
                        val bos = ByteArrayOutputStream()
                        bmp.compress(Bitmap.CompressFormat.JPEG, quality, bos)
                        bmp.recycle()
                        bos.toByteArray()
                    }
                } catch (t: Throwable) {
                    null
                }
                main.post { onJpeg(jpeg) }
            }
        } catch (t: Throwable) { // executor already shut down
            onJpeg(null)
        }
    }

    private fun loadModels(c: SceneConfig) {
        cfg = c
        labels = ctx.assets.open("labels.txt").bufferedReader().readLines().filter { it.isNotBlank() }
        val e = Environment.create()
        env = e
        val dec = YoloeDecoder(labels.size, maskCoefs = c.yoloeMaskCoefs)
        decoder = dec
        yIn = FloatArray(3 * 640 * 640)
        dIn = FloatArray(3 * c.depthSize * c.depthSize)

        val y = ModelRunner(ctx, c.yoloeAsset, Backend.GPU_DEFAULT, e)
        yoloe = y
        y.run(yIn) // also a warm-up; the zero input is fine for finding the output
        for (i in 0 until y.outputCount) if (y.read(i).size == dec.expectedSize) yIdx = i
        check(yIdx >= 0) { "YOLOE: no output of size ${dec.expectedSize}; wrong asset or yoloeMaskCoefs?" }

        val d = ModelRunner(ctx, c.depthAsset, Backend.GPU_DEFAULT, e)
        depth = d
        d.run(dIn)
        val warm = d.read(0)
        check(warm.size == c.depthSize * c.depthSize) { "depth output ${warm.size} != ${c.depthSize * c.depthSize}" }

        latestDepth = null
        frameId = 0
    }

    private fun closeModels() {
        runCatching { yoloe?.close() }
        runCatching { depth?.close() }
        yoloe = null
        depth = null
        latestDepth = null
    }

    private fun bindCamera(ex: ExecutorService) {
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            try {
                if (exec !== ex) return@addListener // stopped while the models were loading
                val p = future.get()
                provider = p
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER),
                            )
                            .build(),
                    )
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) // drop frames while busy
                    .build()
                analysis.setAnalyzer(ex) { image -> analyze(image) }
                readHfov()?.let { hfovDeg = it }
                p.unbindAll()
                p.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            } catch (t: Throwable) {
                fail(t)
            }
        }, ContextCompat.getMainExecutor(ctx))
    }

    private fun analyze(image: ImageProxy) {
        try {
            val c = cfg ?: return
            val dec = decoder ?: return
            val y = yoloe ?: return
            val d = depth ?: return

            val t0 = System.nanoTime()
            conv.convert(image)
            val f = conv.frame
            val t1 = System.nanoTime()

            val doDepth = latestDepth == null || frameId % c.depthEvery == 0L
            val lb = Preprocess.letterbox(f.argb, f.w, f.h, 640, yIn)
            val m1 = System.nanoTime()
            y.run(yIn) // enqueues GPU work and returns
            val m2 = System.nanoTime()
            var depthEnqNs = 0L // depth preprocess + enqueue
            var yReadNs = 0L // waits for YOLOE on the GPU
            var decodeNs = 0L
            val dets: List<Detection>
            if (c.overlap) {
                // CPU preprocessing for depth overlaps the GPU running YOLOE; decode overlaps GPU running depth.
                if (doDepth) {
                    val a = System.nanoTime()
                    Preprocess.stretch(f.argb, f.w, f.h, c.depthSize, dIn)
                    d.run(dIn)
                    depthEnqNs = System.nanoTime() - a
                }
                val a = System.nanoTime()
                val yo = y.read(yIdx)
                yReadNs = System.nanoTime() - a
                val b = System.nanoTime()
                dets = dec.decode(yo, lb, c.confThr, c.iouThr)
                decodeNs = System.nanoTime() - b
            } else {
                val a = System.nanoTime()
                val yo = y.read(yIdx)
                yReadNs = System.nanoTime() - a
                val b = System.nanoTime()
                dets = dec.decode(yo, lb, c.confThr, c.iouThr)
                decodeNs = System.nanoTime() - b
                if (doDepth) {
                    val e = System.nanoTime()
                    Preprocess.stretch(f.argb, f.w, f.h, c.depthSize, dIn)
                    d.run(dIn)
                    depthEnqNs = System.nanoTime() - e
                }
            }
            var depthReadNs = 0L // waits for depth on the GPU
            if (doDepth) {
                val a = System.nanoTime()
                latestDepth = d.read(0)
                depthReadNs = System.nanoTime() - a
                latestDepthNs = System.nanoTime()
            }
            val t2 = System.nanoTime()

            // ---- fusion ----
            val dm = latestDepth
            val objs = ArrayList<HashMap<String, Any?>>(dets.size)
            for (det in dets) {
                val x1 = det.x1 / f.w
                val y1 = det.y1 / f.h
                val x2 = det.x2 / f.w
                val y2 = det.y2 / f.h
                val o = HashMap<String, Any?>()
                o["label"] = labels[det.classId]
                o["score"] = det.score.toDouble()
                o["box"] = listOf(x1.toDouble(), y1.toDouble(), x2.toDouble(), y2.toDouble())
                if (dm != null) {
                    val s = DepthSampler.boxStats(dm, c.depthSize, x1, y1, x2, y2, scratch)
                    o["depthMedian"] = s.median.toDouble()
                    o["depthP10"] = s.p10.toDouble()
                }
                objs.add(o)
            }
            val t3 = System.nanoTime()

            val ev = HashMap<String, Any?>()
            ev["frameId"] = frameId
            ev["tMs"] = image.imageInfo.timestamp / 1e6
            ev["w"] = f.w
            ev["h"] = f.h
            ev["detections"] = objs
            ev["depthFresh"] = doDepth
            ev["depthAgeMs"] = if (dm != null) (t3 - latestDepthNs) / 1e6 else -1.0
            if (dm != null) {
                ev["gridRows"] = GRID_ROWS
                ev["gridCols"] = GRID_COLS
                ev["grid"] = DepthSampler.gridStats(dm, c.depthSize, GRID_ROWS, GRID_COLS, scratch)
            }
            ev["hfovDeg"] = hfovDeg.toDouble()
            if (doDepth && dm != null) { // a fresh depth map: hand the planner its grid
                DepthSampler.downsample(dm, c.depthSize, GRID_W, GRID_H, gridBuf, scratch)
                ev["depthGrid"] = gridBuf.copyOf()
                ev["dgW"] = GRID_W
                ev["dgH"] = GRID_H
            }
            if (viz) ev["previewJpeg"] = previewJpeg(f)
            val t4 = System.nanoTime()
            ev["timing"] = hashMapOf<String, Any?>(
                "convMs" to (t1 - t0) / 1e6,
                "modelsMs" to (t2 - t1) / 1e6, // preprocess + both models + decode, as overlapped
                "fuseMs" to (t4 - t2) / 1e6,
                "totalMs" to (t4 - t0) / 1e6,
                // breakdown of modelsMs
                "yPreMs" to (m1 - t1) / 1e6, // YOLOE letterbox
                "yRunMs" to (m2 - m1) / 1e6, // YOLOE enqueue
                "dEnqMs" to depthEnqNs / 1e6, // depth preprocess + enqueue
                "yReadMs" to yReadNs / 1e6, // wait for YOLOE
                "decodeMs" to decodeNs / 1e6,
                "dReadMs" to depthReadNs / 1e6, // wait for depth
            )
            frameId++
            main.post { onFrame(ev) }
        } catch (t: Throwable) {
            fail(t)
        } finally {
            image.close()
        }
    }

    private fun fail(t: Throwable) {
        main.post {
            onError("${t.javaClass.simpleName}: ${t.message}")
            stop()
        }
    }

    private companion object {
        const val GRID_ROWS = 3
        const val GRID_COLS = 5
        const val GRID_W = 48 // planner depth grid (columns x rows of the upright frame)
        const val GRID_H = 64
        const val PREVIEW_W = 192
        const val PREVIEW_H = 256
    }
}
