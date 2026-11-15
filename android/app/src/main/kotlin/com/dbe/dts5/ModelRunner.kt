package com.dbe.dts5

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.Closeable

enum class Backend { CPU, GPU_DEFAULT, GPU_FP32 }

/**
 * Thin wrapper over LiteRT's CompiledModel.
 *
 * UNVERIFIED API SURFACES (written from documentation fragments, never compiled):
 *  - Environment.create()                      -> how an Environment is built in Kotlin
 *  - CompiledModel.Options(setOf(Accelerator)) + gpuOptions = GpuOptions(precision = Precision.FP32)
 *    (seen in a model card using LiteRT 2.2.0; may differ in 2.3.0)
 * If the build fails, the error will point at one of these lines.
 *
 * GPU failures are NOT retried on CPU by LiteRT, so callers must catch and fall back themselves.
 */
class ModelRunner(
    context: Context,
    asset: String,
    val backend: Backend,
    env: Environment,
) : Closeable {

    private val model: CompiledModel = CompiledModel.create(context.assets, asset, optionsFor(backend), env)
    private val inputs = model.createInputBuffers()
    private val outputs = model.createOutputBuffers()

    var lastRunMs: Double = 0.0
        private set

    val outputCount: Int get() = outputs.size

    /** Writes the input and executes. Reading outputs is separate so unneeded outputs are never copied. */
    fun run(input: FloatArray) {
        val t0 = System.nanoTime()
        inputs[0].writeFloat(input)
        model.run(inputs, outputs)
        lastRunMs = (System.nanoTime() - t0) / 1e6
    }

    fun read(index: Int): FloatArray = outputs[index].readFloat()

    override fun close() {
        runCatching { inputs.forEach { it.close() } }
        runCatching { outputs.forEach { it.close() } }
        runCatching { model.close() }
    }

    private companion object {
        fun optionsFor(b: Backend): CompiledModel.Options = when (b) {
            Backend.CPU -> CompiledModel.Options(Accelerator.CPU)
            Backend.GPU_DEFAULT -> CompiledModel.Options(Accelerator.GPU)
            Backend.GPU_FP32 -> CompiledModel.Options(setOf(Accelerator.GPU)).apply {
                gpuOptions = CompiledModel.GpuOptions(
                    precision = CompiledModel.GpuOptions.Precision.FP32,
                )
            }
        }
    }
}
