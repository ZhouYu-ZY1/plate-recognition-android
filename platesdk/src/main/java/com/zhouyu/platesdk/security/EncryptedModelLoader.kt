package com.zhouyu.platesdk.security

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.zhouyu.platesdk.config.PlateConfig
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

object EncryptedModelLoader {

    private val magic = byteArrayOf(0x5a, 0x50, 0x4d, 0x31)
    private const val VERSION: Byte = 1
    private const val HEADER_SIZE = 5
    private const val IV_SIZE = 12
    private const val TAG_BITS = 128

    fun createSession(
        context: Context,
        environment: OrtEnvironment,
        assetName: String,
        sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
    ): OrtSession {
        val encrypted = context.assets.open("${PlateConfig.ASSETS_DIR}/$assetName").use { it.readBytes() }
        val key = ModelKeyProvider.getModelKey(context.applicationContext)
        val modelBytes = decrypt(encrypted, key)

        return try {
            environment.createSession(modelBytes, sessionOptions)
        } finally {
            modelBytes.fill(0)
            key.fill(0)
        }
    }

    private fun decrypt(encrypted: ByteArray, key: ByteArray): ByteArray {
        require(encrypted.size > HEADER_SIZE + IV_SIZE) { "Invalid encrypted model size" }
        require(encrypted.copyOfRange(0, magic.size).contentEquals(magic)) { "Invalid encrypted model header" }
        require(encrypted[magic.size] == VERSION) { "Unsupported encrypted model version" }

        val ivStart = HEADER_SIZE
        val cipherStart = ivStart + IV_SIZE
        val iv = encrypted.copyOfRange(ivStart, cipherStart)
        val cipherText = encrypted.copyOfRange(cipherStart, encrypted.size)

        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, SecretKeySpec(key, "AES"), GCMParameterSpec(TAG_BITS, iv))
        return cipher.doFinal(cipherText)
    }
}
