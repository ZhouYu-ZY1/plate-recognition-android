package com.zhouyu.platesdk

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import com.zhouyu.platesdk.config.PlateConfig
import com.zhouyu.platesdk.model.PlateResult
import com.zhouyu.platesdk.recognition.PlateColorClassifier
import com.zhouyu.platesdk.recognition.PlateDetector
import com.zhouyu.platesdk.recognition.PlateRecognizer
import com.zhouyu.platesdk.security.EncryptedModelLoader
import com.zhouyu.platesdk.utils.ImageUtils
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * 车牌检测识别 SDK 统一入口（单例）
 *
 * 使用示例：
 * ```kotlin
 * PlateRecognitionSDK.init(context)
 * val results = PlateRecognitionSDK.recognize(bitmap)
 * PlateRecognitionSDK.release()
 * ```
 */
object PlateRecognitionSDK {

    private var ortEnv: OrtEnvironment? = null
    private var detectSession: OrtSession? = null
    private var recSession: OrtSession? = null
    private var isInitialized = false
    private val lock = ReentrantLock()

    /**
     * 初始化 SDK：从加密 assets 解密模型并在内存中创建 ONNX Session
     */
    fun init(context: Context): Boolean = lock.withLock {
        if (isInitialized) return@withLock true

        try {
            // 加载 OpenCV native 库
            if (!org.opencv.android.OpenCVLoader.initLocal()) {
                System.loadLibrary("opencv_java4")
            }

            // 创建 ONNX Runtime 环境
            ortEnv = OrtEnvironment.getEnvironment()
            val environment = ortEnv ?: return@withLock false

            // 从加密模型创建检测 Session，不落地明文模型文件
            val detectSo = OrtSession.SessionOptions()
            detectSession = EncryptedModelLoader.createSession(
                context = context,
                environment = environment,
                assetName = PlateConfig.DETECT_MODEL_ASSET,
                sessionOptions = detectSo
            )

            // 从加密模型创建识别 Session，不落地明文模型文件
            val recSo = OrtSession.SessionOptions()
            recSession = EncryptedModelLoader.createSession(
                context = context,
                environment = environment,
                assetName = PlateConfig.REC_MODEL_ASSET,
                sessionOptions = recSo
            )

            isInitialized = true
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    /**
     * 完整车牌检测识别流水线（包含颜色识别）
     *
     * 流程：检测 → 透视变换 → 文字识别 + 颜色识别 → 合并结果
     *
     * @param bitmap 输入图像（支持任意尺寸）
     * @return 识别结果列表（已包含号牌文字和颜色）
     */
    fun recognize(bitmap: Bitmap): List<PlateResult> = lock.withLock {
        if (!isInitialized) {
            throw IllegalStateException("SDK 未初始化，请先调用 PlateRecognitionSDK.init(context)")
        }

        val results = mutableListOf<PlateResult>()

        try {
            // Bitmap → Mat (RGBA → BGR)
            val srcMat = Mat()
            Utils.bitmapToMat(bitmap, srcMat)
            val bgrMat = Mat()
            Imgproc.cvtColor(srcMat, bgrMat, Imgproc.COLOR_RGBA2BGR)
            srcMat.release()

            // 车牌检测
            val detResults = PlateDetector.detect(detectSession!!, bgrMat)

            for (detBox in detResults) {
                // 透视变换矫正车牌（fourPointTransform 接受 FloatArray）
                val landmarks = detBox.landmarks
                val warped = ImageUtils.fourPointTransform(bgrMat, landmarks)

                // 文字识别
                val isDouble = detBox.plateType == "双层"
                val plateText = PlateRecognizer.recognize(
                    recSession!!, warped, isDouble
                )

                // 颜色识别（在透视变换后的车牌图上进行，与 C# 一致）
                val plateColor = PlateColorClassifier.classify(warped)

                results.add(detBox.copy(plateText = plateText, plateColor = plateColor))

                warped.release()
            }

            bgrMat.release()
        } catch (e: Exception) {
            e.printStackTrace()
        }

        results
    }

    /**
     * 释放所有 ONNX 资源
     */
    fun release() = lock.withLock {
        try {
            detectSession?.close()
            recSession?.close()
            ortEnv?.close()
        } catch (_: Exception) {
        } finally {
            detectSession = null
            recSession = null
            ortEnv = null
            isInitialized = false
        }
    }
}
