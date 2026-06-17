package com.zhouyu.platesdk.utils

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * 文件工具类：负责从 Assets 拷贝模型到内部存储
 */
object FileUtils {

    /**
     * 将 assets/platesdk 下的 .onnx 文件拷贝到 cacheDir/platesdk/
     * @return 拷贝后的目录路径
     */
    fun copyModelsFromAssets(context: Context, assetsSubDir: String = "platesdk"): String {
        val targetDir = File(context.cacheDir, assetsSubDir)
        if (!targetDir.exists()) {
            targetDir.mkdirs()
        }
        Log.e("File", "copyModelsFromAssets: "+targetDir.absolutePath)

        val assetFiles = context.assets.list(assetsSubDir) ?: emptyArray()

        for (fileName in assetFiles) {
            Log.e("File", "copyModelsFromAssets: $targetDir/$fileName")
            val targetFile = File(targetDir, fileName)
            if (targetFile.exists()) continue  // 已拷贝，跳过

            try {
                Log.e("File", "1copyModelsFromAssets: $assetsSubDir/$fileName")
                context.assets.open("$assetsSubDir/$fileName").use { input ->
                    FileOutputStream(targetFile).use { output ->
                        Log.e("File", "2copyModelsFromAssets: $assetsSubDir/$fileName")
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                throw RuntimeException("Failed to copy $fileName: ${e.message}")
            }
        }

        return targetDir.absolutePath
    }

    /**
     * 获取模型文件的完整路径
     */
    fun getModelPath(context: Context, modelFileName: String, assetsSubDir: String = "platesdk"): String {
        return File(context.cacheDir, "$assetsSubDir/$modelFileName").absolutePath
    }
}
