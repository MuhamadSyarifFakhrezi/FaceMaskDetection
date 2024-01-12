package com.example.facemaskdetection

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.facemaskdetection.ml.Maskdetection
import com.example.facemaskdetection.ui.theme.FaceMaskDetectionTheme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : ComponentActivity() {

    lateinit var selectbtn: Button
    lateinit var predictbtn: Button
    lateinit var resView: TextView
    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main_activity)

        selectbtn = findViewById(R.id.selectbtn)
        predictbtn = findViewById(R.id.predictbtn)
        resView = findViewById(R.id.resView)
        imageView = findViewById(R.id.imageView)

        var labels = application.assets.open("label.txt").bufferedReader().readLines()



        selectbtn.setOnClickListener {
            var intent = Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(intent, 100)
        }

        predictbtn.setOnClickListener {

            // image processor
            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)

            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(128, 128, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .build()

            tensorImage = imageProcessor.process(tensorImage)

            val model = Maskdetection.newInstance(this)

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 128, 128, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(tensorImage.buffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

//            var maxIdx = 0
//            outputFeature0.forEachIndexed { index, fl ->
//                if(outputFeature0[maxIdx] < fl){
//                    maxIdx = index
//                }
//            }

            // Ambil label berdasarkan nilai probabilitas
            val predictedLabel = getLabelFromProbability(outputFeature0[0])

            // Tampilkan label di UI
            resView.text = predictedLabel
//            resView.setText(labels[maxIdx])

            Log.d("Debug", "Output Feature: ${outputFeature0.contentToString()}")


// Releases model resources if no longer used.
            model.close()
        }
    }

    private fun getLabelFromProbability(probability: Float): String {
        val threshold = 0.5 // Sesuaikan ambang batas sesuai kebutuhan

        return if (probability > threshold) {
            "Without Mask" // Jika probabilitas lebih besar dari ambang batas
        } else {
            "With Mask" // Jika probabilitas kurang dari atau sama dengan ambang batas
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if(requestCode == 100) {
            var uri = data?.data;
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imageView.setImageBitmap(bitmap)
        }
    }
}