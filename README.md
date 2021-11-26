# age-and-gender-detection
Objective :

To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.

The contents of this Project :

opencv_face_detector.pbtxt

opencv_face_detector_uint8.pb

age_deploy.prototxt

age_net.caffemodel

gender_deploy.prototxt

gender_net.caffemodel

a few pictures to try the project on

AgeGender.py

Models

Download models from

Gender Net : https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=0"

Age Net : https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=0"


For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.
