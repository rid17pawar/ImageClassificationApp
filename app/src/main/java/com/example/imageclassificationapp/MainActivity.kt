package com.example.imageclassificationapp

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import com.example.imageclassificationapp.ml.MobilenetV110224Quant
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.util.*
import kotlin.collections.ArrayList

/*
* This app is developed using TensorflowLite pretrained model: mobilenet_v1_1.0_224_quant_and_labels
* Add the labels file in assets folder, as well as add tflite model in project by File->New->Other->TensorflowLite Model
*
* Model download Link: https://www.tensorflow.org/lite/examples/image_classification/overview
* Youtube video Link: https://youtu.be/6ErbFQb8QS8
*
* For testing, Download images from internet whose names are available in assets/labels_mobilenet_dataset.txt file.
*/

class MainActivity : AppCompatActivity() {
    val TAG:String= "MainActivity"
    lateinit var bitmap_img: Bitmap
    lateinit var labels_list: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        //Load labels list from labels_mobilenet_dataset file in assets folder
        try{
            val labels_reader= BufferedReader(InputStreamReader(application.assets.open(("labels_mobilenet_dataset.txt"))))
            labels_list= labels_reader.readLines()

            Log.d(TAG, "labels list="+labels_list.get(0))
        }catch(e: Exception){
            Log.d(TAG, "exception in reading labels from file: "+e)
        }

        uploadbtn.setOnClickListener {
            //change textview data
            detailstxt.text="Press Search button to get details of the above Image."

            try{
                //Pick image from gallery
                var intent1= Intent(Intent.ACTION_PICK)
                intent1.type="image/*"
                //after choosing an image, gallery will close
                startActivityForResult(intent1, 100) //override this method to provide custom behaviour
            }catch(e: Exception){
                Log.d(TAG, "exception in creating intent for image upload: "+e)
            }
        }

        searchbtn.setOnClickListener {
            //check if image is uploaded by user
            if(bitmap_img==null){
                detailstxt.text= "Please upload an image first then click on search button"
            }

            //resize uploadedimage
            var resized_bitmap_img: Bitmap= Bitmap.createScaledBitmap(bitmap_img, 224,224, true)

            //paste the tflite model code here
            try{
                val model = MobilenetV110224Quant.newInstance(this)

                // Creates inputs for reference.
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)

            //create bytebuffer image from the resized bitmap
                var byteBuffer= TensorImage.fromBitmap(resized_bitmap_img).buffer
                inputFeature0.loadBuffer(byteBuffer)

                // Runs model inference and gets result.
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            //show output in textView of layout file
                //outputFeature0 array has probabilities of 1000 values
                var maxProbabilityIndex:Int= getMaxProbabilityIndex(outputFeature0.floatArray)
                Log.d(TAG, "returned maxProbabilityIndex="+maxProbabilityIndex)

                var resulttxt= "item name: "+labels_list[maxProbabilityIndex]+", probability: "+outputFeature0.floatArray[maxProbabilityIndex]
                detailstxt.text= resulttxt

                // Releases model resources if no longer used.
                model.close()
            }catch(e: Exception){
                Log.d(TAG, "exception in using model for predictions: "+e)
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        try{
            //when user selects the image, it will be replaced in imageView.
            if(resultCode == Activity.RESULT_OK && requestCode == 100){
                image1.setImageURI(data?.data)

                //store uploaded image in bitmap variable. Image data is passed in data variable of this function.
                var selected_img_uri:Uri?= data?.data
                bitmap_img= MediaStore.Images.Media.getBitmap(this.contentResolver, selected_img_uri)
            }
            else{
                Toast.makeText(this, "Unable to preview image", Toast.LENGTH_SHORT)
            }

        }catch(e: Exception){
            Log.d(TAG, "exception in storing image in bitmap: "+e)
        }
    }

    fun getMaxProbabilityIndex(arr: FloatArray): Int {
        var max_val: Float= 0.0F
        var max_val_index:Int= -1
        var items_count_max_probalities:Int=0

        for(i in 0..1000) {
            //find max probability item index
            if(arr[i]>max_val) {
                max_val_index=i
                max_val=arr[i]
            }
            //find no. of items with and above 100 probability
            if(arr[i]>=100){
                items_count_max_probalities++
            }
        }
        Log.d(TAG, "no. of items with probability >=100 are "+items_count_max_probalities)
        return max_val_index
    }

}