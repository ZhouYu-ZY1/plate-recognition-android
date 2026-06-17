package com.zhouyu.plate

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.zhouyu.plate.ui.theme.PlaterecognitionandroidTheme
import com.zhouyu.platesdk.PlateRecognitionSDK
import com.zhouyu.platesdk.model.PlateResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            PlaterecognitionandroidTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    PlateRecognitionScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@Composable
fun PlateRecognitionScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var sourceBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var resultBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var results by remember { mutableStateOf<List<PlateResult>>(emptyList()) }
    var time by remember { mutableLongStateOf(0L) }
    var isProcessing by remember { mutableStateOf(false) }

    // 初始化 SDK
    var sdkReady by remember { mutableStateOf(false) }

    // 图库选择器
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            try {
                val inputStream = context.contentResolver.openInputStream(it)
                sourceBitmap = BitmapFactory.decodeStream(inputStream)
                resultBitmap = null
                results = emptyList()
                inputStream?.close()
            } catch (e: Exception) {
                Toast.makeText(context, "读取图片失败: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // 执行识别
    fun doRecognize() {
        val bmp = sourceBitmap ?: return
        if (!sdkReady) {
            sdkReady = PlateRecognitionSDK.init(context)
            if (!sdkReady) {
                Toast.makeText(context, "SDK 初始化失败", Toast.LENGTH_SHORT).show()
                return
            }
        }

        val startTime = System.currentTimeMillis()
        isProcessing = true
        scope.launch {
            try {
                val recognizeResults = withContext(Dispatchers.Default) {
                    PlateRecognitionSDK.recognize(bmp)
                }
                time = System.currentTimeMillis() - startTime
                Log.e("scope", "doRecognize: "+time)
                results = recognizeResults
                // 在原图上绘制检测框
                resultBitmap = drawResults(bmp, recognizeResults)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(context, "识别失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } finally {
                isProcessing = false
            }
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "车牌检测识别",
            style = MaterialTheme.typography.headlineMedium
        )

        Spacer(modifier = Modifier.height(16.dp))

        // 操作按钮
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Button(
                onClick = { galleryLauncher.launch("image/*") },
                modifier = Modifier.weight(1f)
            ) {
                Text("选择图片")
            }

            Button(
                onClick = { doRecognize() },
                modifier = Modifier.weight(1f),
                enabled = sourceBitmap != null && !isProcessing
            ) {
                Text(if (isProcessing) "识别中..." else "开始识别")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 原图预览
        sourceBitmap?.let { bmp ->
            Text("原图", style = MaterialTheme.typography.titleMedium)
            Spacer(modifier = Modifier.height(8.dp))
            Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = "原图",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(16.dp))
        }

        // 结果图
        resultBitmap?.let { bmp ->
            Text("识别结果", style = MaterialTheme.typography.titleMedium)
            Spacer(modifier = Modifier.height(8.dp))
            Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = "结果图",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(16.dp))
        }

        // 文字结果列表
        if (results.isNotEmpty()) {
            Text("识别详情", style = MaterialTheme.typography.titleMedium)
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "[推理耗时：${time}ms]",
                style = MaterialTheme.typography.bodyMedium
            )
            results.forEachIndexed { index, result ->
                Text(
                    text = "(${index + 1}) ${result.plateText} ${result.plateColor}  (${result.plateType})  置信度: ${"%.2f".format(result.detectScore)}\n" +
                            "坐标:(${"%.0f".format(result.x1)},${"%.0f".format(result.y1)})(${"%.0f".format(result.x2)},${"%.0f".format(result.y2)})",
                    style = MaterialTheme.typography.bodyMedium
                )
                Spacer(modifier = Modifier.height(4.dp))
            }
        }
    }
}

/**
 * 根据车牌颜色返回对应的绘制颜色
 */
private fun getColorForPlate(plateColor: String): Int {
    return when (plateColor) {
        "蓝色" -> Color.BLUE
        "黄色" -> Color.rgb(255, 200, 0)
        "白色" -> Color.WHITE
        "绿色" -> Color.rgb(0, 180, 0)
        "黄绿色" -> Color.rgb(128, 200, 0)
        else -> Color.RED
    }
}

/**
 * 在原图上绘制检测框和号牌文字
 */
private fun drawResults(src: Bitmap, results: List<PlateResult>): Bitmap {
    val bitmap = src.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(bitmap)

    for (result in results) {
        val plateDrawColor = getColorForPlate(result.plateColor)

        // 检测框画笔（颜色跟随车牌颜色）
        val rectPaint = Paint().apply {
            color = plateDrawColor
            style = Paint.Style.STROKE
            strokeWidth = 4f
            isAntiAlias = true
        }

        val rect = RectF(result.x1, result.y1, result.x2, result.y2)
        canvas.drawRect(rect, rectPaint)

        // 绘制四个角点
        val cornerPaint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        val landmarks = result.landmarks
        for (i in 0 until 4) {
            canvas.drawCircle(landmarks[i * 2], landmarks[i * 2 + 1], 6f, cornerPaint)
        }

        // 文字背景画笔
        val textBgPaint = Paint().apply {
            color = plateDrawColor
            style = Paint.Style.FILL
        }

        // 文字画笔
        val textPaint = Paint().apply {
            // 蓝色/绿色背景用白字，黄色/白色背景用黑字
            color = when (result.plateColor) {
                "黄色", "白色" -> Color.BLACK
                else -> Color.WHITE
            }
            textSize = 36f
            isAntiAlias = true
            isFakeBoldText = true
        }

        // 绘制号牌文字（包含颜色信息）
        val text = "${result.plateText} (${result.plateColor}) ${"%.2f".format(result.detectScore)}"
        val textWidth = textPaint.measureText(text)
        canvas.drawRect(
            result.x1,
            result.y1 - 40f,
            result.x1 + textWidth + 8f,
            result.y1,
            textBgPaint
        )
        canvas.drawText(text, result.x1 + 4f, result.y1 - 8f, textPaint)
    }

    return bitmap
}
