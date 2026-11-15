package com.dbe.dts5

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Build
import android.os.Process
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.tanh

/** One continuous cue stream, as sent from Dart. */
class CueParams(val pan: Float, val ppm: Float, val hz: Float, val gain: Float, val locked: Boolean)

/**
 * Real-time cue synthesiser: a dedicated thread streams stereo float audio, so pulse timing is precise
 * and independent of the 3-7 Hz vision frame rate. Parameters arrive from Dart; pan and gain are smoothed.
 * Two continuous streams (hazard, beacon) plus one-shot earcons. If no update arrives for STALE_NS the
 * streams fade out, so a stalled camera can never leave a tone stuck on.
 *
 * Plays with USAGE_ASSISTANCE_NAVIGATION_GUIDANCE / CONTENT_TYPE_SONIFICATION so the system can route and
 * duck it correctly. Stereo pan only means something on headphones, not on the phone speaker.
 */
class CueSynth {
    private companion object {
        const val SR = 48000
        const val BLOCK = 256 // frames per write (~5 ms)
        const val STALE_NS = 700_000_000L
        const val TWO_PI = (2 * PI).toFloat()
        val EARCONS = mapOf(
            "found" to floatArrayOf(660f, 880f),
            "arrived" to floatArrayOf(660f, 830f, 990f, 1320f),
            "lost" to floatArrayOf(660f, 440f),
            "alert" to floatArrayOf(520f, 520f, 520f),
            "listen_on" to floatArrayOf(520f, 780f),
            "listen_off" to floatArrayOf(780f, 520f),
        )
    }

    @Volatile private var hazards: Array<CueParams?> = arrayOfNulls(3) // left, centre, right
    @Volatile private var beacon: CueParams? = null
    @Volatile private var hazardAtNs = 0L
    @Volatile private var beaconAtNs = 0L
    @Volatile private var duckTarget = 1f
    private val earconQueue = ConcurrentLinkedQueue<String>()

    @Volatile private var running = false
    private var thread: Thread? = null
    private var track: AudioTrack? = null

    fun update(h: List<CueParams?>, b: CueParams?) {
        val now = System.nanoTime()
        hazards = Array(3) { h.getOrNull(it) }
        hazardAtNs = now
        beacon = b
        beaconAtNs = now
    }

    fun earcon(name: String) {
        if (EARCONS.containsKey(name)) earconQueue.add(name)
    }

    /** 1 = normal, 0 = silent: applied to the beacon and earcons. Hazard tones are never ducked. */
    fun setDuck(level: Float) {
        duckTarget = level.coerceIn(0f, 1f)
    }

    @Synchronized
    fun start() {
        if (running) return
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()
        val fmt = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setSampleRate(SR)
            .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
            .build()
        val minBytes = AudioTrack.getMinBufferSize(SR, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_FLOAT)
        val builder = AudioTrack.Builder()
            .setAudioAttributes(attrs)
            .setAudioFormat(fmt)
            .setBufferSizeInBytes(maxOf(minBytes, BLOCK * 2 * 4 * 2))
            .setTransferMode(AudioTrack.MODE_STREAM)
        if (Build.VERSION.SDK_INT >= 26) builder.setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
        val t = builder.build()
        track = t
        running = true
        t.play()
        thread = Thread({ renderLoop(t) }, "cue-synth").also { it.start() }
    }

    @Synchronized
    fun stop() {
        running = false
        runCatching { thread?.join(500) }
        thread = null
        track?.let {
            runCatching { it.stop() }
            it.release()
        }
        track = null
        hazards = arrayOfNulls(3)
        beacon = null
        earconQueue.clear()
    }

    /** A pulse train (or continuous tone) with a percussive attack, like a sonar / parking sensor. */
    private class Voice(private val tauSec: Float, private val attackSec: Float) {
        var gain = 0f
        var pan = 0f
        var untilNext = 0
        var age = Int.MAX_VALUE
        var accentAt = -1
        var freq = 440f
        var phase = 0f
        var continuous = false // last params were a continuous tone: fade it out instead of cutting it

        /** Adds this voice into [out] (interleaved stereo). [p] == null means silent. */
        fun render(p: CueParams?, n: Int, out: FloatArray, level: Float) {
            val attack = (attackSec * SR).toInt().coerceAtLeast(1)
            val tau = tauSec * SR
            for (i in 0 until n) {
                gain += ((if (p != null) p.gain * level else 0f) - gain) * 0.0015f
                if (p != null) {
                    pan += (p.pan - pan) * 0.0008f
                    continuous = p.ppm <= 0f
                }
                if (gain < 1e-4f && p == null) {
                    age = Int.MAX_VALUE
                    untilNext = 0
                    continue
                }
                if (p != null && p.ppm > 0f) {
                    if (untilNext <= 0) { // a new pulse starts
                        age = 0
                        freq = p.hz
                        untilNext = (60f / p.ppm * SR).toInt()
                        accentAt = if (p.locked) (0.09f * SR).toInt() else -1
                    }
                    untilNext--
                    if (accentAt in 1..age && age >= accentAt) { // "locked" = a second, higher pulse
                        age = 0
                        freq = p.hz * 1.5f
                        accentAt = -1
                    }
                }
                val env = when {
                    p == null -> if (continuous) 1f else 0f // gain smoothing fades a continuous tone out
                    p.ppm <= 0f -> 1f // continuous
                    age == Int.MAX_VALUE -> 0f
                    age < attack -> age.toFloat() / attack
                    else -> exp(-(age - attack) / tau)
                }
                if (age != Int.MAX_VALUE && age < Int.MAX_VALUE - 1) age++
                phase += TWO_PI * freq / SR
                if (phase > TWO_PI) phase -= TWO_PI
                val s = sin(phase) * env * gain
                val a = (pan.coerceIn(-1f, 1f) + 1f) * (PI.toFloat() / 4f) // constant-power pan
                out[2 * i] += s * kotlin.math.cos(a)
                out[2 * i + 1] += s * sin(a)
            }
        }
    }

    private fun renderLoop(t: AudioTrack) {
        runCatching { Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO) }
        val buf = FloatArray(BLOCK * 2)
        val hazardVoices = Array(3) { Voice(tauSec = 0.045f, attackSec = 0.003f) }
        val beaconVoice = Voice(tauSec = 0.09f, attackSec = 0.004f)
        var duckLevel = 1f

        // earcon state: a short arpeggio of decaying notes at the centre
        var notes: FloatArray? = null
        var noteIdx = 0
        var noteAge = 0
        var phase = 0f
        val noteLen = (0.11f * SR).toInt()

        while (running) {
            java.util.Arrays.fill(buf, 0f)
            val now = System.nanoTime()
            val hs = hazards
            val hFresh = now - hazardAtNs < STALE_NS
            val b = beacon?.takeIf { now - beaconAtNs < STALE_NS }
            duckLevel += (duckTarget - duckLevel) * 0.02f

            for (k in 0 until 3) hazardVoices[k].render(if (hFresh) hs[k] else null, BLOCK, buf, 1f)
            beaconVoice.render(b, BLOCK, buf, duckLevel)

            for (i in 0 until BLOCK) {
                if (notes == null) {
                    val name = earconQueue.poll() ?: break
                    notes = EARCONS[name]
                    noteIdx = 0
                    noteAge = 0
                }
                val ns = notes ?: break
                val env = if (noteAge < 144) noteAge / 144f else exp(-(noteAge - 144) / (0.07f * SR))
                phase += TWO_PI * ns[noteIdx] / SR
                if (phase > TWO_PI) phase -= TWO_PI
                val s = sin(phase) * env * 0.5f * maxOf(duckLevel, 0.4f)
                buf[2 * i] += s * 0.7071f
                buf[2 * i + 1] += s * 0.7071f
                if (++noteAge >= noteLen) {
                    noteAge = 0
                    if (++noteIdx >= ns.size) notes = null
                }
            }

            for (i in buf.indices) buf[i] = tanh(buf[i]) // soft clip: summed streams must never wrap or click
            val written = t.write(buf, 0, buf.size, AudioTrack.WRITE_BLOCKING)
            if (written < 0) break
        }
    }
}
