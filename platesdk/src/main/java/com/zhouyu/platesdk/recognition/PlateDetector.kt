package com.zhouyu.platesdk.recognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import com.zhouyu.platesdk.config.PlateConfig
import com.zhouyu.platesdk.model.PlateResult
import com.zhouyu.platesdk.utils.ImageUtils
import com.zhouyu.platesdk.utils.LetterboxInfo
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/**
 * 车牌检测器
 *
 * 1. 调用 ImageUtils 进行 Letterbox 预处理
 * 2. ONNX Runtime 推理
 * 3. 后处理：置信度过滤 → NMS → 坐标还原 → 构建 PlateResult 列表
 */
object PlateDetector {

    /**
     * 检测图像中的车牌区域
     *
     * @param session ONNX 检测模型 Session
     * @param srcMat 输入图像（BGR格式 Mat）
     * @param inputName 模型输入节点名
     * @param outputName 模型输出节点名
     * @return 检测到的车牌列表（xyxy + 4角点 + 置信度 + 类别），坐标已还原到原图
     */
    fun detect(
        session: OrtSession,
        srcMat: Mat,
        inputName: String = PlateConfig.DETECT_INPUT_NAME,
        outputName: String = PlateConfig.DETECT_OUTPUT_NAME
    ): List<PlateResult> {
        // 预处理：Letterbox + 归一化（仅 /255，不减均值不除标准差）
        val (paddedMat, letterboxInfo) = ImageUtils.preprocessDetect(srcMat)
        val inputTensor = matToDetectCHWFloatTensor(paddedMat)
        paddedMat.release()

        // ONNX 推理
        val inputs = mapOf(inputName to inputTensor)
        val outputs = session.run(inputs)
        val rawOutput = outputs.get(outputName).get() as OnnxTensor

        // 动态获取输出维度
        val outputShape = rawOutput.info.shape
        val numAnchors = outputShape[1].toInt()
        val numValues = outputShape[2].toInt()   // 应为 16: cx,cy,w,h,objConf,lx0,ly0,lx1,ly1,lx2,ly2,lx3,ly3,cls0,cls1,(padding?)

        val rawArray = rawOutput.floatBuffer.array()
        inputTensor.close()
        rawOutput.close()

        val confThresh = PlateConfig.DETECT_CONF_THRESHOLD
        val iouThresh = PlateConfig.DETECT_IOU_THRESHOLD

        // 置信度过滤
        val candidateBoxes = mutableListOf<FloatArray>()
        for (i in 0 until numAnchors) {
            val offset = i * numValues
            // obj 置信度（第5个值，索引4）
            val objConf = rawArray[offset + 4]
            if (objConf <= confThresh) continue

            // 类别分 × obj 置信度（cls0在索引13, cls1在索引14）
            val cls0 = rawArray[offset + 13] * objConf
            val cls1 = rawArray[offset + 14] * objConf
            val maxScore = max(cls0, cls1)
            if (maxScore < confThresh) continue

            val label = if (cls0 >= cls1) 0f else 1f
            val cx = rawArray[offset]
            val cy = rawArray[offset + 1]
            val w = rawArray[offset + 2]
            val h = rawArray[offset + 3]

            val x1 = cx - w / 2
            val y1 = cy - h / 2
            val x2 = cx + w / 2
            val y2 = cy + h / 2

            // 四个角点（在 640×640 坐标系下，从 index 5 开始）
            val lx0 = rawArray[offset + 5]
            val ly0 = rawArray[offset + 6]
            val lx1 = rawArray[offset + 7]
            val ly1 = rawArray[offset + 8]
            val lx2 = rawArray[offset + 9]
            val ly2 = rawArray[offset + 10]
            val lx3 = rawArray[offset + 11]
            val ly3 = rawArray[offset + 12]

            candidateBoxes.add(floatArrayOf(
                x1, y1, x2, y2, maxScore,
                lx0, ly0, lx1, ly1, lx2, ly2, lx3, ly3,
                label
            ))
        }

        if (candidateBoxes.isEmpty()) return emptyList()

        // NMS
        val nmsResult = nms(candidateBoxes, iouThresh)

        // 坐标还原到原图 + 构建结果
        val results = mutableListOf<PlateResult>()
        for (box in nmsResult) {
            val restored = restoreBox(box, letterboxInfo)
            results.add(restored)
        }

        return results
    }

    /**
     * 将归一化后的坐标还原到原图坐标系
     */
    private fun restoreBox(box: FloatArray, info: LetterboxInfo): PlateResult {
        val ratio = info.ratio
        val padLeft = info.padLeft.toFloat()
        val padTop = info.padTop.toFloat()
        val origW = info.origW
        val origH = info.origH

        val x1 = clip((box[0] - padLeft) / ratio, 0f, origW)
        val y1 = clip((box[1] - padTop) / ratio, 0f, origH)
        val x2 = clip((box[2] - padLeft) / ratio, 0f, origW)
        val y2 = clip((box[3] - padTop) / ratio, 0f, origH)

        val score = box[4]

        // 四个角点还原到原图
        val corners = floatArrayOf(
            clip((box[5] - padLeft) / ratio, 0f, origW), clip((box[6] - padTop) / ratio, 0f, origH),
            clip((box[7] - padLeft) / ratio, 0f, origW), clip((box[8] - padTop) / ratio, 0f, origH),
            clip((box[9] - padLeft) / ratio, 0f, origW), clip((box[10] - padTop) / ratio, 0f, origH),
            clip((box[11] - padLeft) / ratio, 0f, origW), clip((box[12] - padTop) / ratio, 0f, origH)
        )

        val plateType = if (box[13] == 0f) "单层" else "双层"

        return PlateResult(
            x1 = x1, y1 = y1, x2 = x2, y2 = y2,
            detectScore = score,
            landmarks = corners,
            plateText = "",
            plateType = plateType,
            detectTimeMs = 0L,
            recTimeMs = 0L
        )
    }

    /**
     * NMS（非极大值抑制）
     */
    private fun nms(boxes: List<FloatArray>, iouThresh: Float): List<FloatArray> {
        val sorted = boxes.sortedByDescending { it[4] }.toMutableList()
        val result = mutableListOf<FloatArray>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)

            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val box = iterator.next()
                if (computeIoU(best, box) > iouThresh) {
                    iterator.remove()
                }
            }
        }

        return result
    }

    /**
     * 计算两个框的 IoU
     */
    private fun computeIoU(boxA: FloatArray, boxB: FloatArray): Float {
        val ax1 = boxA[0]; val ay1 = boxA[1]; val ax2 = boxA[2]; val ay2 = boxA[3]
        val bx1 = boxB[0]; val by1 = boxB[1]; val bx2 = boxB[2]; val by2 = boxB[3]

        val interX1 = max(ax1, bx1)
        val interY1 = max(ay1, by1)
        val interX2 = min(ax2, bx2)
        val interY2 = min(ay2, by2)
        val interW = max(0f, interX2 - interX1)
        val interH = max(0f, interY2 - interY1)
        val interArea = interW * interH

        val areaA = (ax2 - ax1) * (ay2 - ay1)
        val areaB = (bx2 - bx1) * (by2 - by1)
        val union = areaA + areaB - interArea

        return if (union <= 1e-6f) 0f else interArea / union
    }

    private fun clip(value: Float, min: Float, max: Float) = value.coerceIn(min, max)

    /**
     * Mat (BGR, 8UC3) → CHW 归一化 float 数组 → OnnxTensor
     *
     * 检测模型预处理：BGR→RGB + /255 归一化，不减均值不除标准差
     * 使用批量像素读取优化性能（避免逐像素 JNI 调用）
     */
    private fun matToDetectCHWFloatTensor(mat: Mat): OnnxTensor {
        val height = mat.rows()
        val width = mat.cols()
        val channels = 3

        // 批量获取所有像素数据，避免逐像素 JNI 调用（性能优化：从 ~123万次 JNI 调用降为 1 次）
        val rgbMat = Mat()
        Imgproc.cvtColor(mat, rgbMat, Imgproc.COLOR_BGR2RGB)
        val floatMat = Mat()
        rgbMat.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)
        rgbMat.release()

        val totalPixels = height * width * channels
        val hwcData = FloatArray(totalPixels)
        floatMat.get(0, 0, hwcData)
        floatMat.release()

        // HWC → CHW 转换
        val floats = FloatArray(channels * height * width)
        for (h in 0 until height) {
            for (w in 0 until width) {
                val hwcIdx = (h * width + w) * channels
                // 已经是 RGB 顺序
                floats[0 * height * width + h * width + w] = hwcData[hwcIdx]     // R
                floats[1 * height * width + h * width + w] = hwcData[hwcIdx + 1] // G
                floats[2 * height * width + h * width + w] = hwcData[hwcIdx + 2] // B
            }
        }

        return OnnxTensor.createTensor(
            ai.onnxruntime.OrtEnvironment.getEnvironment(),
            java.nio.FloatBuffer.wrap(floats),
            longArrayOf(1, channels.toLong(), height.toLong(), width.toLong())
        )
    }
}
