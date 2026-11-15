package com.dbe.dts5

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.view.KeyEvent
import android.view.WindowManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

/**
 * If the build says "Type mismatch ... LifecycleOwner" for `this` below, change the base class to
 * FlutterFragmentActivity (same package, io.flutter.embedding.android) and tell me.
 */
class MainActivity : FlutterActivity() {
    private var engine: SceneEngine? = null
    private val synth = CueSynth()
    private val voices = mapOf(
        "guidance" to VoicePlayer(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE, AudioAttributes.CONTENT_TYPE_SPEECH),
        "gemini" to VoicePlayer(AudioAttributes.USAGE_ASSISTANT, AudioAttributes.CONTENT_TYPE_SPEECH),
    )
    private var micSink: EventChannel.EventSink? = null
    private val mic = MicCapture { bytes -> runOnUiThread { micSink?.success(bytes) } }
    private var pendingMicResult: MethodChannel.Result? = null
    private var keySink: EventChannel.EventSink? = null
    private var captureVolumeKeys = false
    private var sink: EventChannel.EventSink? = null
    private var pendingConfig: Map<*, *>? = null
    private var pendingResult: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger

        // ---- benchmark (unchanged; needs Bench.kt v3 in the same package) ----
        MethodChannel(messenger, "spatial_nav/bench").setMethodCallHandler { call, result ->
            when (call.method) {
                "run" -> Thread {
                    val text = try {
                        Bench(applicationContext).run()
                    } catch (t: Throwable) {
                        "FATAL: $t\n${t.stackTraceToString()}"
                    }
                    runOnUiThread { result.success(text) }
                }.start()
                "last" -> result.success(Bench.lastReport(applicationContext))
                else -> result.notImplemented()
            }
        }

        // ---- sonification cues ----
        MethodChannel(messenger, "spatial_nav/cues").setMethodCallHandler { call, result ->
            when (call.method) {
                "start" -> {
                    synth.start()
                    result.success(null)
                }
                "stop" -> {
                    synth.stop()
                    result.success(null)
                }
                "update" -> {
                    val m = call.arguments as Map<*, *>
                    synth.update(((m["hazards"] as? List<*>) ?: emptyList<Any?>()).map { parseCue(it) }, parseCue(m["beacon"]))
                    (m["earcons"] as? List<*>)?.forEach { (it as? String)?.let(synth::earcon) }
                    result.success(null)
                }
                "duck" -> {
                    synth.setDuck((call.arguments as Number).toFloat())
                    result.success(null)
                }
                "earcon" -> {
                    synth.earcon(call.arguments as String)
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        }

        // ---- voices (Kokoro guidance clips, Gemini's streamed voice) ----
        MethodChannel(messenger, "spatial_nav/voice").setMethodCallHandler { call, result ->
            val v = call.argument<String>("voice")?.let { voices[it] }
            if (v == null) {
                result.error("voice", "unknown voice", null)
                return@setMethodCallHandler
            }
            when (call.method) {
                "start" -> {
                    v.start()
                    result.success(null)
                }
                "stop" -> {
                    v.stop()
                    result.success(null)
                }
                "enqueue" -> {
                    val pcm = call.argument<ByteArray>("pcm")
                    if (pcm == null) {
                        result.error("voice", "missing pcm", null)
                    } else if (call.argument<Boolean>("await") == true) {
                        v.enqueue(pcm) { runOnUiThread { result.success(null) } } // completes when played or flushed
                    } else {
                        v.enqueue(pcm)
                        result.success(null)
                    }
                }
                "flush" -> {
                    v.flush()
                    result.success(null)
                }
                "pendingMs" -> result.success(v.pendingMs())
                else -> result.notImplemented()
            }
        }

        // ---- microphone ----
        MethodChannel(messenger, "spatial_nav/mic").setMethodCallHandler { call, result ->
            when (call.method) {
                "start" -> startMic(result)
                "stop" -> {
                    mic.stop()
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        }
        EventChannel(messenger, "spatial_nav/mic_events").setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink) {
                micSink = events
            }

            override fun onCancel(arguments: Any?) {
                micSink = null
            }
        })

        EventChannel(messenger, "spatial_nav/keys").setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink) {
                keySink = events
            }

            override fun onCancel(arguments: Any?) {
                keySink = null
            }
        })

        // ---- scene engine ----
        engine = SceneEngine(
            applicationContext, this,
            onFrame = { sink?.success(it) },
            onError = { sink?.error("engine", it, null) },
        )
        MethodChannel(messenger, "spatial_nav/engine").setMethodCallHandler { call, result ->
            when (call.method) {
                "start" -> startEngine(call.arguments as Map<*, *>, result)
                "stop" -> {
                    stopEngine()
                    result.success(null)
                }
                "viz" -> {
                    engine?.setViz(call.arguments as Boolean)
                    result.success(null)
                }
                // Volume up/down become Gemini on/off while the engine runs; otherwise they change the volume as usual.
                "captureVolumeKeys" -> {
                    captureVolumeKeys = call.arguments as Boolean
                    result.success(null)
                }
                "snapshot" -> {
                    val e = engine
                    if (e == null) result.success(null) else e.snapshot { result.success(it) }
                }
                else -> result.notImplemented()
            }
        }
        EventChannel(messenger, "spatial_nav/engine_events").setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink) {
                sink = events
            }

            override fun onCancel(arguments: Any?) {
                sink = null
            }
        })
    }

    private fun isVolumeKey(keyCode: Int) = keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (captureVolumeKeys && isVolumeKey(keyCode)) {
            if ((event?.repeatCount ?: 0) == 0) keySink?.success(if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) "up" else "down")
            return true // consumed: no system volume change, no volume panel
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent?): Boolean {
        if (captureVolumeKeys && isVolumeKey(keyCode)) return true
        return super.onKeyUp(keyCode, event)
    }

    private fun startMic(result: MethodChannel.Result) {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            pendingMicResult = result
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_MIC)
            return
        }
        if (mic.start()) result.success(null) else result.error("mic", "could not start the microphone", null)
    }

    private fun parseCue(a: Any?): CueParams? {
        val m = a as? Map<*, *> ?: return null
        return CueParams(
            (m["pan"] as Number).toFloat(),
            (m["ppm"] as Number).toFloat(),
            (m["hz"] as Number).toFloat(),
            (m["gain"] as Number).toFloat(),
            m["locked"] as Boolean,
        )
    }

    private fun startEngine(cfg: Map<*, *>, result: MethodChannel.Result) {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            pendingConfig = cfg
            pendingResult = result
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
            return
        }
        launch(cfg, result)
    }

    private fun launch(cfg: Map<*, *>, result: MethodChannel.Result) {
        try {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) // headless but the screen stays on
            engine?.start(SceneConfig.fromMap(cfg))
            result.success(null)
        } catch (t: Throwable) {
            result.error("start", t.toString(), null)
        }
    }

    private fun stopEngine() {
        engine?.stop()
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_MIC) {
            val res = pendingMicResult
            pendingMicResult = null
            if (res != null) {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    if (mic.start()) res.success(null) else res.error("mic", "could not start the microphone", null)
                } else {
                    res.error("permission", "Microphone permission denied", null)
                }
            }
            return
        }
        if (requestCode != REQ_CAMERA) return
        val cfg = pendingConfig
        val res = pendingResult
        pendingConfig = null
        pendingResult = null
        if (cfg == null || res == null) return
        if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            launch(cfg, res)
        } else {
            res.error("permission", "Camera permission denied", null)
        }
    }

    override fun onDestroy() {
        engine?.stop()
        synth.stop()
        mic.stop()
        voices.values.forEach { it.stop() }
        super.onDestroy()
    }

    private companion object {
        const val REQ_CAMERA = 4711
        const val REQ_MIC = 4712
    }
}
