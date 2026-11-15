package com.dbe.dts5

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.NoiseSuppressor

/**
 * 16 kHz mono PCM16 microphone capture in 40 ms chunks, the format the Live API takes.
 * Uses the VOICE_COMMUNICATION source (platform echo cancellation / noise suppression where the device has it)
 * and additionally enables AcousticEchoCanceler / NoiseSuppressor when available. Without echo cancellation the
 * microphone will hear Kokoro and Gemini through the speaker and Gemini may answer itself: use headphones.
 * Needs the RECORD_AUDIO permission (MainActivity asks for it).
 */
class MicCapture(private val onChunk: (ByteArray) -> Unit) {
    private companion object {
        const val RATE = 16000
        const val CHUNK_BYTES = RATE / 25 * 2 // 40 ms
    }

    private var rec: AudioRecord? = null
    private var aec: AcousticEchoCanceler? = null
    private var ns: NoiseSuppressor? = null
    private var thread: Thread? = null
    @Volatile private var running = false

    @SuppressLint("MissingPermission")
    @Synchronized
    fun start(): Boolean {
        if (running) return true
        val minBuf = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (minBuf <= 0) return false
        val r = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION, RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, RATE / 5 * 2),
        )
        if (r.state != AudioRecord.STATE_INITIALIZED) {
            r.release()
            return false
        }
        if (AcousticEchoCanceler.isAvailable()) aec = AcousticEchoCanceler.create(r.audioSessionId)?.also { it.setEnabled(true) }
        if (NoiseSuppressor.isAvailable()) ns = NoiseSuppressor.create(r.audioSessionId)?.also { it.setEnabled(true) }
        rec = r
        running = true
        r.startRecording()
        thread = Thread({ readLoop(r) }, "mic-capture").also { it.start() }
        return true
    }

    @Synchronized
    fun stop() {
        running = false
        runCatching { thread?.join(500) }
        thread = null
        rec?.let {
            runCatching { it.stop() }
            it.release()
        }
        rec = null
        runCatching { aec?.release() }
        runCatching { ns?.release() }
        aec = null
        ns = null
    }

    private fun readLoop(r: AudioRecord) {
        val chunk = ByteArray(CHUNK_BYTES)
        while (running) {
            var off = 0
            while (off < chunk.size && running) {
                val n = r.read(chunk, off, chunk.size - off)
                if (n < 0) {
                    running = false
                    break
                }
                off += n
            }
            if (off == chunk.size) onChunk(chunk.copyOf())
        }
    }
}
