package com.zhouyu.platesdk.config

/**
 * 全局配置常量
 */
object PlateConfig {

    // ──────────────────────────────────────
    // 号牌字符集
    // ──────────────────────────────────────
    const val PLATE_CHARS = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"

    val CHAR_TO_INDEX: Map<Char, Int> = PLATE_CHARS.withIndex().associate { it.value to it.index }

    // ──────────────────────────────────────
    // 归一化参数
    // ──────────────────────────────────────
    const val MEAN_VALUE: Float = 0.588f
    const val STD_VALUE: Float = 0.193f

    // ──────────────────────────────────────
    // 车牌检测模型参数
    // ──────────────────────────────────────
    const val DETECT_WIDTH: Int = 640
    const val DETECT_HEIGHT: Int = 640
    const val DETECT_CONF_THRESHOLD: Float = 0.4f   // 置信度阈值
    const val DETECT_IOU_THRESHOLD: Float = 0.5f     // NMS IoU阈值
    const val DETECT_INPUT_NAME: String = "input"
    const val DETECT_OUTPUT_NAME: String = "output"

    // ──────────────────────────────────────
    // 车牌识别模型参数
    // ──────────────────────────────────────
    const val REC_WIDTH: Int = 168
    const val REC_HEIGHT: Int = 48
    const val REC_INPUT_NAME: String = "images"

    // ──────────────────────────────────────
    // 双层车牌分割比例
    // ──────────────────────────────────────
    const val DOUBLE_PLATE_CUT_RATIO: Float = 5f / 12f  // 上层高度占比
    const val DOUBLE_PLATE_COMBINE_RATIO: Float = 1f / 3f // 合成时上层缩放到下层宽的比例

    // ──────────────────────────────────────
    // Letterbox 填充颜色（灰色）
    // ──────────────────────────────────────
    val LETTERBOX_COLOR: FloatArray = floatArrayOf(114f, 114f, 114f)

    // ──────────────────────────────────────
    // 加密模型文件
    // ──────────────────────────────────────
    const val ASSETS_DIR: String = "platesdk"
    const val DETECT_MODEL_ASSET: String = "car_plate_detect.onnx.enc"
    const val REC_MODEL_ASSET: String = "plate_rec.onnx.enc"
}
