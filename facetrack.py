import cv2
import numpy as np
from djitellopy import tello
import time
import keyboard

# Setting up the drone
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()


#NETWORK CONFIGURATION
# importing yolov4 cfg and weights
weights = 'yolov4-tiny.weights'
cfg = 'yolov4-tiny.cfg'

# reading to a network
net = cv2.dnn.readNet(weights,cfg)
#########################

# Setting the input size
width = 1280  # WIDTH OF THE IMAGE 640
height = 720  # HEIGHT OF THE IMAGE 480
#########################

# drone controlls options
Range = [188336, 140000]
#140000 new
#188336 new

#old 181699
#old 21676

# settings for PID
pid = [0.2,0,0.2]
preverr = 0
#########################

# fps counter
fps_start_time = 0
fps = 0
########################

# optional for automatic take-off
launched = False

# user input for selecting the objects that is desired to follow
print("Please Enter an Object you want to follow ")
findWord = input("Or press enter to show all objects: ")
# opening and searching the coco.names file
with open('coco.names','r') as f:
    classses = f.read().rstrip('\n').split('\n')
    if findWord in classses:
        print("Word found")
    else:
        print("Word not found, showing a screen")

print(classses)

def findobj(img):
    # creating a blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # getting only the output layer names from YOLO
    outlaynames = net.getUnconnectedOutLayersNames()
    layouts = net.forward(outlaynames)

    boxes = []
    confidences = []
    class_ids = []
    center = []
    arealist = []

    # looping over each of the layer outputs
    for output in layouts:
        # loooping through each detection
        for detection in output:
            # extracting the scores, class ids and confidences
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # filtering out the weak detections
            if confidence > 0.5:
                # scaling the bounding boxes
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                #updating the lists
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    # making sure if at least one object exists
    if len(indexes) > 0:
        # looping through the indexes that we keep
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classses[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            if label.lower() == findWord.lower():
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
                area = w * h
                #print(area)
                arealist.append(area)
                cv2.putText(img, label + " " + confidence, (x, y-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.circle(img, (cx,cy), 5, (0, 255, 0), cv2.FILLED)
                center.append([cx, cy])

    if len(arealist) != 0:
        i = arealist.index(max(arealist))
        return img, [center[i], arealist[i]]
    else:
        return img, [[0, 0], 0]


def track(me,info, width, pid, preError):
    area = info[1]
    x,y = info[0]
    range = 0

    #finding how far away is the object from the center
    error = x-width//2

    #pid
    #diviation is for how far away
    speed = pid[0]*error+pid[2]*(error - preError)
    speed = int(np.clip(speed,-25,25))

    # condition for stationary
    if area > Range[0] and area< Range[1]:
        range = 0
    # too close go back
    elif area > Range[1]:
        range = -20
    #too far - go forward (and makes sure that there is something detected)
    elif area < Range[0] and area !=0:
        range = 20



    if x ==0:
        speed = 0
        error = 0
    #print(speed,range)
    me.send_rc_control(0,range,0,speed)
    return error




# FOR LAPTOP
#cap = cv2.VideoCapture(0)
###########

while True:
    #rec,frame = cap.read()
    img = me.get_frame_read().frame
    ###### FOR LAPTOP
    #rec, img = cap.read()
    #################
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1/(time_diff)
    fps_start_time=fps_end_time
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(img,fps_text,(5,30),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
    img = cv2.resize(img,(width,height))
    ####################################
    img, info = findobj(img)
    preverr = track(info, width, pid, preverr)
    ##################################
    #print("area",info[1],"center",info[0])



    cv2.imshow("Output",img)

    #for automatic launching
    #if launched == False:
        #print("Taking off")
        #me.takeoff()
        #me.send_rc_control(0, 0, 25, 0)
    #launched = True
    if keyboard.is_pressed('w'):  # if key 'q' is pressed
        me.takeoff()
        me.send_rc_control(0,0,80,0)
        print('taking off')



    if cv2.waitKey(1) & 0xFF == ord('q'):
        #me.land()
        break

