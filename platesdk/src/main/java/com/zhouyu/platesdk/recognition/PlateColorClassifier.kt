package com.zhouyu.platesdk.recognition

import com.zhouyu.platesdk.config.PlateConfig
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

/**
 * 车牌颜色分类器
 *
 * 基于 HSV 色彩空间像素统计判断车牌颜色。
 * 纯 OpenCV 实现，
 *
 * 支持识别：蓝色、黄色、白色、绿色、黄绿色（新能源）
 */
object PlateColorClassifier {

    /**
     * 颜色枚举及其对应的 HSV 阈值范围
     */
    enum class PlateColor(
        val label: String,
        val hsvMin: Scalar,
        val hsvMax: Scalar
    ) {
        BLUE("蓝色", Scalar(90.0, 43.0, 46.0), Scalar(124.0, 255.0, 255.0)),
        YELLOW("黄色", Scalar(11.0, 43.0, 46.0), Scalar(34.0, 255.0, 255.0)),
        WHITE("白色", Scalar(0.0, 0.0, 46.0), Scalar(180.0, 30.0, 255.0)),
        GREEN("绿色", Scalar(35.0, 43.0, 46.0), Scalar(89.0, 255.0, 255.0));
    }

    /**
     * 识别车牌图像的主要颜色
     *
     * @param plateImg 车牌区域图像（BGR 格式 Mat，来自透视变换后）
     * @return 车牌颜色名称（蓝色/黄色/白色/绿色/黄绿色）
     */
    fun classify(plateImg: Mat): String {
        //  中值滤波去噪（核大小 7）
        val blurred = Mat()
        Imgproc.medianBlur(plateImg, blurred, 7)

        // BGR → HSV 转换
        val hsv = Mat()
        Imgproc.cvtColor(blurred, hsv, Imgproc.COLOR_BGR2HSV)
        blurred.release()

        // 统计各颜色像素数
        val colorCounts = mutableMapOf<PlateColor, Int>()
        var maxCount = 0
        var maxColor = PlateColor.BLUE

        for (color in PlateColor.values()) {
            val count = countColorPixels(hsv, color.hsvMin, color.hsvMax)
            colorCounts[color] = count
            if (count > maxCount) {
                maxCount = count
                maxColor = color
            }
        }

        hsv.release()

        // 特殊判断：绿色为主但黄色占比 > 30% → 黄绿色（新能源车牌）
        if (maxColor == PlateColor.GREEN && maxCount > 0) {
            val yellowCount = colorCounts[PlateColor.YELLOW] ?: 0
            val rate = yellowCount.toDouble() / maxCount.toDouble()
            if (rate > 0.3) {
                return "黄绿色"
            }
        }

        return maxColor.label
    }

    /**
     * 统计 HSV 图像中指定颜色范围内的像素数量
     * 使用 Core.countNonZero 替代逐像素遍历（性能优化）
     */
    private fun countColorPixels(hsv: Mat, hsvMin: Scalar, hsvMax: Scalar): Int {
        val mask = Mat()
        Core.inRange(hsv, hsvMin, hsvMax, mask)
        val count = Core.countNonZero(mask)
        mask.release()
        return count
    }
}
