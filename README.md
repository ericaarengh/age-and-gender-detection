# age-and-gender-detection
Objective :

To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.

About the Project :

In this Python Project, we had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. 
we used the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, 
and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43),
(48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image
because of factors like makeup, lighting, obstructions, and facial expressions.
And so, we made this a classification problem instead of making it one of regression.

The contents of this Project :

opencv_face_detector.pbtxt

opencv_face_detector_uint8.pb

age_deploy.prototxt

age_net.caffemodel

gender_deploy.prototxt

gender_net.caffemodel

A few pictures to try the project on

AgeGender.py

Models

Downloaded Models from

Gender Net : https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=0"

Age Net : https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=0"


For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.

So, here we have:
-Detected faces

-Classified into Male/Female

-Classified into one of the 8 age ranges

-Put the results on the image and displayed it.

Display in Python

python AgeGender.py --input <input_file>(Leave blank for webcam)

Sample Result

(sample-output.jpg)
