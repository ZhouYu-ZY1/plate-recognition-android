package com.zhouyu.platesdk.utils

import android.graphics.Bitmap
import com.zhouyu.platesdk.config.PlateConfig
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/**
 * 图像预处理工具类
 */
object ImageUtils {

    // ──────────────────────────────────────
    // Letterbox + 检测预处理
    // ──────────────────────────────────────

    /**
     * 检测模型预处理：Letterbox → 归一化 → float数组
     * @return Pair(padded图像Mat, 用于还原坐标的ratio/pad信息)
     */
    fun preprocessDetect(src: Mat): Pair<Mat, LetterboxInfo> {
        // 1. 计算缩放比例和填充
        val origH = src.rows().toFloat()
        val origW = src.cols().toFloat()
        val targetSize = PlateConfig.DETECT_WIDTH.toFloat()

        val ratio = targetSize / maxOf(origW, origH)
        val newW = (origW * ratio).toInt()
        val newH = (origH * ratio).toInt()

        // 2. 缩放
        val resized = Mat()
        Imgproc.resize(src, resized, Size(newW.toDouble(), newH.toDouble()))

        // 3. Letterbox：灰色填充
        val padW = (PlateConfig.DETECT_WIDTH - newW) / 2
        val padH = (PlateConfig.DETECT_HEIGHT - newH) / 2
        val padded = Mat(PlateConfig.DETECT_HEIGHT, PlateConfig.DETECT_WIDTH, CvType.CV_8UC3,
            Scalar(114.0, 114.0, 114.0))

        resized.copyTo(padded.submat(
            padH, padH + newH,
            padW, padW + newW
        ))
        resized.release()

        return Pair(padded, LetterboxInfo(ratio, padW, padH, origW, origH))
    }

    /**
     * 将还原后的坐标裁剪到原图范围内
     */
    fun clipToImage(x: Float, y: Float, imgW: Float, imgH: Float): Pair<Float, Float> {
        return Pair(x.coerceIn(0f, imgW), y.coerceIn(0f, imgH))
    }

    // ──────────────────────────────────────
    // 四点透视变换
    // ──────────────────────────────────────

    /**
     * 角点排序：按左上、右上、右下、左下顺序排列
     */
    private fun orderPoints(pts: Array<Point>): Array<Point> {
        require(pts.size == 4) { "需要四个点" }
        val rect = arrayOfNulls<Point>(4)

        val sums = pts.map { it.x + it.y }
        val diffs = pts.map { it.y - it.x }

        rect[0] = pts[sums.indexOf(sums.min())]   // 左上：x+y 最小
        rect[2] = pts[sums.indexOf(sums.max())]   // 右下：x+y 最大
        rect[1] = pts[diffs.indexOf(diffs.min())] // 右上：y-x 最小
        rect[3] = pts[diffs.indexOf(diffs.max())] // 左下：y-x 最大

        @Suppress("UNCHECKED_CAST")
        return rect as Array<Point>
    }

    /**
     * 根据四个角点做透视变换，获得正视角车牌图像
     * - 使用 OrderPoints 排序角点
     * - 根据角点距离动态计算目标宽高（不固定 168×48）
     *
     * @param src 原始图像
     * @param pts 四个角点坐标 [x0,y0, x1,y1, x2,y2, x3,y3]
     */
    fun fourPointTransform(src: Mat, pts: FloatArray): Mat {
        // 将 FloatArray 转为 Point 数组
        val points = arrayOf(
            Point(pts[0].toDouble(), pts[1].toDouble()),
            Point(pts[2].toDouble(), pts[3].toDouble()),
            Point(pts[4].toDouble(), pts[5].toDouble()),
            Point(pts[6].toDouble(), pts[7].toDouble())
        )

        // 排序角点
        val rect = orderPoints(points)
        val tl = rect[0]  // 左上
        val tr = rect[1]  // 右上
        val br = rect[2]  // 右下
        val bl = rect[3]  // 左下

        // 根据角点距离计算目标宽高
        val widthA = sqrt((br.x - bl.x).pow(2) + (br.y - bl.y).pow(2))
        val widthB = sqrt((tr.x - tl.x).pow(2) + (tr.y - tl.y).pow(2))
        val maxWidth = max(widthA.toInt(), widthB.toInt()).coerceAtLeast(1)

        val heightA = sqrt((tr.x - br.x).pow(2) + (tr.y - br.y).pow(2))
        val heightB = sqrt((tl.x - bl.x).pow(2) + (tl.y - bl.y).pow(2))
        val maxHeight = max(heightA.toInt(), heightB.toInt()).coerceAtLeast(1)

        val srcPoints = MatOfPoint2f(tl, tr, br, bl)
        val dstPoints = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(maxWidth.toDouble() - 1, 0.0),
            Point(maxWidth.toDouble() - 1, maxHeight.toDouble() - 1),
            Point(0.0, maxHeight.toDouble() - 1)
        )

        val matrix = Imgproc.getPerspectiveTransform(srcPoints, dstPoints)
        val result = Mat()
        Imgproc.warpPerspective(src, result, matrix, Size(maxWidth.toDouble(), maxHeight.toDouble()))
        matrix.release()
        return result
    }

    // ──────────────────────────────────────
    // 双层车牌拆分与拼接
    // ──────────────────────────────────────

    /**
     * 双层车牌处理：将上层裁出并缩放拼接到下层右边
     * - 上层: [0, 5/12*h]
     * - 下层: [1/3*h, h]
     * - 上层缩放到下层的 width × height
     */
    fun splitMerge(plateMat: Mat): Mat {
        val h = plateMat.rows()
        val w = plateMat.cols()

        // 上层高度（与 C# 一致：5/12 * h）
        val upperHeight = (5f / 12f * h).toInt()
        // 下层起始位置（与 C# 一致：1/3 * h）
        val lowerStart = (1f / 3f * h).toInt()

        if (upperHeight <= 0 || lowerStart >= h) return plateMat.clone()

        val upperPart = Mat(plateMat, Rect(0, 0, w, upperHeight)).clone()
        val lowerPart = Mat(plateMat, Rect(0, lowerStart, w, h - lowerStart)).clone()

        if (lowerPart.empty() || upperPart.empty()) {
            upperPart.release()
            lowerPart.release()
            return plateMat.clone()
        }

        // 上层缩放到下层的 width × height（与 C# 一致）
        val scaledUpper = Mat()
        Imgproc.resize(upperPart, scaledUpper, Size(lowerPart.cols().toDouble(), lowerPart.rows().toDouble()))
        upperPart.release()

        // 水平拼接：上层缩放后 + 下层
        val combined = Mat()
        Core.hconcat(listOf(scaledUpper, lowerPart), combined)
        lowerPart.release()
        scaledUpper.release()

        return combined
    }

    // ──────────────────────────────────────
    // 识别模型预处理
    // ──────────────────────────────────────

    /**
     * 识别模型预处理：resize → 归一化 → float数组（CHW格式）
     * @param src 校正后的车牌图像
     * @return FloatArray 形状 [1, 3, 48, 168]，已归一化
     */
    fun preprocessRec(src: Mat, targetW: Int = PlateConfig.REC_WIDTH, targetH: Int = PlateConfig.REC_HEIGHT): FloatArray {
        // 1. Resize
        val resized = Mat()
        Imgproc.resize(src, resized, Size(targetW.toDouble(), targetH.toDouble()))

        // 2. BGR → RGB + 归一化
        val pixels = FloatArray(3 * targetH * targetW)
        val channels = 3
        for (c in 0 until channels) {
            for (h in 0 until targetH) {
                for (w in 0 until targetW) {
                    val pixel = resized.get(h, w)
                    // OpenCV 读取为 BGR 顺序，需要转为 RGB
                    val bgrIndex = 2 - c  // c=0→R(取BGR[2]), c=1→G(取BGR[1]), c=2→B(取BGR[0])
                    val value = pixel[bgrIndex].toFloat() / 255f
                    val normalized = (value - PlateConfig.MEAN_VALUE) / PlateConfig.STD_VALUE
                    pixels[c * targetH * targetW + h * targetW + w] = normalized
                }
            }
        }

        resized.release()
        return pixels
    }

    // ──────────────────────────────────────
    // 工具方法
    // ──────────────────────────────────────

    /**
     * Bitmap → Mat (BGR)
     */
    fun bitmapToMat(bitmap: Bitmap): Mat {
        val mat = Mat()
        val bmp = bitmap.copy(Bitmap.Config.ARGB_8888, false)
        Utils.bitmapToMat(bmp, mat)
        // Android Bitmap 为 RGBA，转 BGR
        val bgr = Mat()
        Imgproc.cvtColor(mat, bgr, Imgproc.COLOR_RGBA2BGR)
        mat.release()
        return bgr
    }

    /**
     * Mat (BGR) → Bitmap
     */
    fun matToBitmap(mat: Mat): Bitmap {
        val rgba = Mat()
        Imgproc.cvtColor(mat, rgba, Imgproc.COLOR_BGR2RGBA)
        val bitmap = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bitmap)
        rgba.release()
        return bitmap
    }
}

/**
 * Letterbox 变换信息，用于还原坐标到原图
 */
data class LetterboxInfo(
    /** 缩放比例 */
    val ratio: Float,
    /** 左侧填充宽度 */
    val padLeft: Int,
    /** 顶部填充高度 */
    val padTop: Int,
    /** 原始图像宽度 */
    val origW: Float,
    /** 原始图像高度 */
    val origH: Float
)
