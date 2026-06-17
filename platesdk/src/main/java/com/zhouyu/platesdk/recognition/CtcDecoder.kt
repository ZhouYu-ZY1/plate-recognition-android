package com.zhouyu.platesdk.recognition

import com.zhouyu.platesdk.config.PlateConfig

/**
 * CTC（连接时序分类）贪心解码器
 *
 * - 跳过 blank(索引0) 字符
 * - 合并连续重复的相同字符
 * - 过滤非法索引
 */
object CtcDecoder {

    /**
     * 对识别模型输出进行 CTC 贪心解码
     *
     * @param indices 每个时间步 argmax 后的字符索引数组
     * @return 解码后的号牌字符串
     */
    fun decode(indices: IntArray): String {
        val chars = PlateConfig.PLATE_CHARS
        var previous = 0
        val sb = StringBuilder()

        for (idx in indices) {
            // 跳过 blank、跳过与上一个相同的字符、跳过非法索引
            if (idx != 0 && idx != previous && idx < chars.length) {
                sb.append(chars[idx])
            }
            previous = idx
        }

        return sb.toString()
    }
}
