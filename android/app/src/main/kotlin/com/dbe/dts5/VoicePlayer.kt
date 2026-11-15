package com.dbe.dts5

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Build
import android.os.Process
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Plays 16-bit mono PCM through a persistent low-latency AudioTrack. Used twice: one instance for the
 * pre-rendered Kokoro guidance clips (navigation-guidance usage) and one for Gemini's streamed voice
 * (assistant usage), so the OS can route and duck them separately.
 *
 * - enqueue(pcm, onDone): onDone fires once the audio has (by wall-clock estimate) finished playing, or as
 *   soon as it is flushed. Exactly once per item.
 * - flush(): drops everything queued and everything in the track buffer immediately (barge-in, hazards).
 * - pendingMs(): audio still to be heard, for drain detection.
 */
class VoicePlayer(
    private val usage: Int,
    private val contentType: Int,
    private val sampleRate: Int = 24000,
) {
    private class Item(val pcm: ByteArray, val onDone: (() -> Unit)?)

    private val queue = LinkedBlockingQueue<Item>()
    private val generation = AtomicInteger(0)
    @Volatile private var running = false
    @Volatile private var playEndNs = 0L
    private var thread: Thread? = null
    private var track: AudioTrack? = null
    private val trackLock = Any()

    @Synchronized
    fun start() {
        if (running) return
        val attrs = AudioAttributes.Builder().setUsage(usage).setContentType(contentType).build()
        val fmt = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(sampleRate)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()
        val minBytes = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val builder = AudioTrack.Builder()
            .setAudioAttributes(attrs)
            .setAudioFormat(fmt)
            .setBufferSizeInBytes(maxOf(minBytes, sampleRate / 5 * 2)) // at least 200 ms
            .setTransferMode(AudioTrack.MODE_STREAM)
        if (Build.VERSION.SDK_INT >= 26) builder.setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
        val t = builder.build()
        track = t
        running = true
        t.play()
        thread = Thread({ loop() }, "voice-player").also { it.start() }
    }

    @Synchronized
    fun stop() {
        running = false
        runCatching { thread?.join(500) }
        thread = null
        val drained = ArrayList<Item>()
        queue.drainTo(drained)
        drained.forEach { it.onDone?.invoke() }
        track?.let {
            runCatching { it.stop() }
            it.release()
        }
        track = null
    }

    fun enqueue(pcm: ByteArray, onDone: (() -> Unit)? = null) {
        if (!running) {
            onDone?.invoke()
            return
        }
        queue.put(Item(pcm, onDone))
    }

    fun flush() {
        generation.incrementAndGet()
        val drained = ArrayList<Item>()
        queue.drainTo(drained)
        drained.forEach { it.onDone?.invoke() }
        synchronized(trackLock) {
            track?.let { runCatching { it.pause(); it.flush(); it.play() } }
        }
        playEndNs = 0L
    }

    fun pendingMs(): Long {
        val playing = maxOf(0L, (playEndNs - System.nanoTime()) / 1_000_000)
        val queued = queue.sumOf { it.pcm.size / 2 * 1000L / sampleRate }
        return playing + queued
    }

    private fun loop() {
        runCatching { Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO) }
        val piece = sampleRate / 50 * 2 // 20 ms per write, so a flush takes effect within ~20 ms
        while (running) {
            val item = queue.poll(50, TimeUnit.MILLISECONDS) ?: continue
            val gen = generation.get()
            var off = 0
            var aborted = false
            while (off < item.pcm.size && running) {
                if (generation.get() != gen) {
                    aborted = true
                    break
                }
                val n = minOf(piece, item.pcm.size - off)
                val w = track?.write(item.pcm, off, n) ?: -1 // blocks while the track buffer is full
                if (w < 0) {
                    aborted = true
                    break
                }
                if (w == 0) {
                    Thread.sleep(2)
                    continue
                }
                off += w
                playEndNs = maxOf(playEndNs, System.nanoTime()) + w / 2 * 1_000_000_000L / sampleRate
            }
            if (!aborted && item.onDone != null) {
                while (running && generation.get() == gen && System.nanoTime() < playEndNs) Thread.sleep(5)
            }
            item.onDone?.invoke()
        }
    }
}
