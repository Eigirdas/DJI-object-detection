# DJI-object-detection
Object tracking with DJI tello edu drone using YOLO algorithm
Code and source files:
For the source files, there is only one single python file with all the code included for
detection and drone controls, however, since a YOLO V4 object detection algorithm is being
used, additional files included such as coco.names, that have all the available classes, a
configuration file that describes each layer of the network and a weight file, that is used to
store binary values that make a full bulk of the neural network.
The code that was produced is divided into these 6 sections: the first being just the overall
options and values, the second part is for the selection of the objects that should be
highlighted in detection and given bounding boxes, the third part is for the actual algorithm
for finding the objects and it’s options, the fourth part is for drone controls, the fifth part is
the main loop, and the sixth part is for user choice of when the drone should lift off from the
ground as well as a mini safety feature for the landing.
Starting from imports, one of the main is cv2, which is an import name of a module for
OpenCV-python. It is a huge open-source library mainly used in computer vision, machine
learning, and image processing tasks with included several hundreds of computer vision
algorithms. The numpy import is also important because this library is used mainly for
working with arrays, which was later on. For the next import, there is a djitellopy import,
which is a library used for the drone commands and unlocking full controls of the drone,
import of time for calculating FPS, and finally import of keyboard, which allows to use
keyboard and assign buttons to do functions.
Next a variable is set to invoke a setting for calling drone functions and giving a full access to
controls and commands, also connecting the drone (which I need to connect manually with
the drone via WiFi) and then the drone connects to my laptop, so it is ready to listen for
commands, also printing the battery life just for general checking and preparing the drone
camera with streamon command. After that the YOLO weights and cfg into variables and a
deep learning network is read. Afterwards, an image size is set, or in my case, camera size
input, it is set by the value manually to 720p (1280x720) since this is the resolution of the
drones’ camera, It can also be changed to 640x480 which is my laptop webcam resolution.
The FPS variables are then set and counted afterward and for an optional feature, there is
an included automatic drone launch after there is a camera rolling, since it takes time to
boot and launch the object detection algorithm as well as load the neural network It can be
done it manually with a keyboard button. Also, as mentioned before, there has been added
a possibility to follow not one, but multiple-choice objects, that is from 80 available options,
a command opens the coco.names file, parses it in classes list and sets a variable that can be
manually set by the users input, if the user input word was found in coco name file, this
object will be found and put a bounding box, but if not, no bounding boxes will be added.
Finding the object
After that, a function called findobj() is defined, a blob is created, which is a 4-dimensional
numpy array object that accepts values such as image, a scale factor (which is 1/255), the 
