# SDK release 混淆规则

-dontwarn ai.onnxruntime.**
-dontwarn org.opencv.**

-keep class ai.onnxruntime.** { *; }
-keep class org.opencv.** { *; }

-keep class com.zhouyu.platesdk.PlateRecognitionSDK {
    public *;
}

-keep class com.zhouyu.platesdk.model.PlateResult {
    public *;
}

-keepattributes *Annotation*
-renamesourcefileattribute SourceFile
