package com.zhouyu.platesdk.recognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import com.zhouyu.platesdk.config.PlateConfig
import com.zhouyu.platesdk.utils.ImageUtils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Core
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer

/**
 * 车牌文字识别器
 *
 * 1. 透视变换矫正车牌
 * 2. 双层车牌拆分合并
 * 3. resize + 归一化
 * 4. ONNX Runtime 推理
 * 5. CTC 贪心解码
 */
object PlateRecognizer {

    /**
     * 识别车牌区域文字
     *
     * @param session ONNX 识别模型 Session
     * @param roiImg 车牌区域 ROI 图像（BGR格式 Mat，来自透视变换后）
     * @param isDoubleLayer 是否为双层车牌（双层时执行 splitMerge）
     * @param inputName 模型输入节点名
     * @return 识别出的号牌字符串
     */
    fun recognize(
        session: OrtSession,
        roiImg: Mat,
        isDoubleLayer: Boolean = false,
        inputName: String = PlateConfig.REC_INPUT_NAME
    ): String {
        // 复制 ROI 避免修改原图
        var processed = roiImg.clone()

        // 双层车牌拆分合并
        if (isDoubleLayer) {
            val merged = try {
                ImageUtils.splitMerge(processed)
            } catch (e: Exception) {
                null
            }
            processed.release()
            processed = merged ?: roiImg.clone()
        }

        // resize
        val resized = Mat()
        Imgproc.resize(processed, resized, org.opencv.core.Size(
            PlateConfig.REC_WIDTH.toDouble(), PlateConfig.REC_HEIGHT.toDouble()
        ))
        processed.release()

        // 转 float 并归一化: 先 to float /255, 再减均值除以标准差
        val floatMat = Mat()
        resized.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)
        resized.release()

        // 减均值除标准差
        val meanMat = Mat(PlateConfig.REC_HEIGHT, PlateConfig.REC_WIDTH, CvType.CV_32FC3,
            Scalar(PlateConfig.MEAN_VALUE.toDouble(), PlateConfig.MEAN_VALUE.toDouble(), PlateConfig.MEAN_VALUE.toDouble()))
        val stdMat = Mat(PlateConfig.REC_HEIGHT, PlateConfig.REC_WIDTH, CvType.CV_32FC3,
            Scalar(PlateConfig.STD_VALUE.toDouble(), PlateConfig.STD_VALUE.toDouble(), PlateConfig.STD_VALUE.toDouble()))
        val normMat = Mat()
        Core.subtract(floatMat, meanMat, floatMat)
        Core.divide(floatMat, stdMat, normMat)
        floatMat.release()
        meanMat.release()
        stdMat.release()

        // Mat → CHW float 数组 → OnnxTensor（保持 BGR 通道顺序）
        // 使用批量像素读取优化性能（避免逐像素 JNI 调用）
        val inputTensor = matToCHWBGRFloatTensorBatch(normMat)
        normMat.release()

        // ONNX 推理
        val inputs = mapOf(inputName to inputTensor)
        val outputs = session.run(inputs)
        val rawOutput = outputs.first().value as OnnxTensor

        // 输出形状: [1, seqLen, numClasses]
        val shape = rawOutput.info.shape // [batch, seqLen, numClasses]
        val seqLen = shape[1].toInt()
        val numClasses = shape[2].toInt()

        val rawArray = rawOutput.floatBuffer.array()
        inputTensor.close()
        rawOutput.close()

        // argmax per time step
        val predIndices = IntArray(seqLen)
        for (i in 0 until seqLen) {
            var maxIdx = 0
            var maxVal = rawArray[i * numClasses + 0]
            for (c in 1 until numClasses) {
                val value = rawArray[i * numClasses + c]
                if (value > maxVal) {
                    maxVal = value
                    maxIdx = c
                }
            }
            predIndices[i] = maxIdx
        }

        // CTC 贪心解码
        return CtcDecoder.decode(predIndices)
    }

    /**
     * Mat (BGR, 32FC3) → CHW float 数组 → OnnxTensor（保持 BGR 通道顺序）
     * 使用批量像素读取优化性能（避免逐像素 JNI 调用）
     *
     * 输入通道顺序: B, G, R
     */
    private fun matToCHWBGRFloatTensorBatch(mat: Mat): OnnxTensor {
        val height = mat.rows()
        val width = mat.cols()
        val channels = 3

        // 批量获取所有像素数据（1 次 JNI 调用替代 height*width 次）
        val totalPixels = height * width * channels
        val hwcData = FloatArray(totalPixels)
        mat.get(0, 0, hwcData)

        // HWC → CHW 转换（保持 BGR 顺序）
        val floats = FloatArray(channels * height * width)
        for (h in 0 until height) {
            for (w in 0 until width) {
                val hwcIdx = (h * width + w) * channels
                // BGR 顺序: c=0→B, c=1→G, c=2→R
                floats[0 * height * width + h * width + w] = hwcData[hwcIdx]     // B
                floats[1 * height * width + h * width + w] = hwcData[hwcIdx + 1] // G
                floats[2 * height * width + h * width + w] = hwcData[hwcIdx + 2] // R
            }
        }

        return OnnxTensor.createTensor(
            ai.onnxruntime.OrtEnvironment.getEnvironment(),
            FloatBuffer.wrap(floats),
            longArrayOf(1, channels.toLong(), height.toLong(), width.toLong())
        )
    }
}
