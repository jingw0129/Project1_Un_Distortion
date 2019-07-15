# Project1_Un_Distortion
Image_Un_Distortion_python_OpenCV
A project for image unditorton based on funtionality in OpenCV and python as programming language.
loading video files by OpenCV.
Read each frame, which is going to be gone through the algorithm and get the corrected x,y index.
Building a map which is used for storing index of the flattern array(frame array). So we don't have to calculate it next time.
So the map is the index chart we use every time whenever wanna do image unditortion.
The algorithm is based on a paper.
When new image array comes in, we find the poxel based on the ele in map matrix. and save the ele and position into a new frame, which is considered as unditorition image.
new frame is always, definitely corresponding to the map/.
The series of untortion arrays are going to use for line detection.
