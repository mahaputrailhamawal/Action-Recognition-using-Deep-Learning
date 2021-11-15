import os
import sys
import cv2
import time
import argparse
import datetime
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime

# Import OpenPose Library.
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../../python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
import pyopenpose as op

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_pose"] = "COCO"
params["net_resolution"] = "160x-1"
params["model_folder"] = "../../../models/"

# Set Parameters
input_size = (100, 100) # You can change it according to the input size from the training model

# Define input shape
channel = (3,)
input_shape = input_size + channel
labels = ['Sitting', 'Standing', 'Walking']

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

# Import Keras and load training model
from tensorflow.keras.models import load_model
MODEL_PATH = 'F:/Putra/Coba/openpose/build/examples/tutorial_api_python/model100baru(7).h5'
model = load_model(MODEL_PATH,compile=False)

label_lama = "pertama"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Interface Camera
capture = cv2.VideoCapture(0)
datum = op.Datum()

def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))/255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)

while True:
    ret, img = capture.read()
    datum.cvInputData = img
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    
    # Checking skeleton data and set the bounding box from skeleton data
    if datum.poseKeypoints is not None:
        x_awal = datum.poseKeypoints[0][3][0]
        x_akhir = datum.poseKeypoints[0][6][0]
        
        y_awal1 = datum.poseKeypoints[0][16][1]
        y_awal2 = datum.poseKeypoints[0][17][1]
        if y_awal1 > y_awal2:
            y_awal = y_awal1
        else:
            y_awal = y_awal2
        
        y_akhir1 = datum.poseKeypoints[0][10][1]
        y_akhir2 = datum.poseKeypoints[0][13][1]
        if y_akhir1 > y_akhir2:
            y_akhir = y_akhir1
        else:
            y_akhir = y_akhir2
        if x_awal and x_akhir and y_awal and y_akhir > 0:
            if x_awal < x_akhir:
                crop_img = datum.cvOutputData[int(y_awal-40):int(y_akhir+50), int(x_awal-50):int(x_akhir+50)]
            else:
                crop_img = datum.cvOutputData[int(y_awal-40):int(y_akhir+50), int(x_akhir-50):int(x_awal+50)]

            # Action prediction
            im = Image.fromarray(crop_img)
            cX = preprocess(im,input_size)
            cX = reshape([cX])
            cY = model.predict(cX)

            # Draw bounding box from action recognition
            cv2.rectangle(img, (x_awal, y_awal), (x_akhir, y_akhir), (255, 0, 0), 3)
            cv2.putText(img,f'{labels[np.argmax(cY)]} {int(np.max(cY)*100)}%',
                      (x_awal, y_awal), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
    
    # Merge real-time frame and the output action recognition in one frame
    skel = datum.cvOutputData
    final_output = cv2.addWeighted(skel, 1, img, 1, 0)
    
    # font which we will be using to display FPS 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # time when we finish processing for this frame 
    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
  
    # converting the fps into integer 
    fps = int(fps) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps = str(fps) 
  
    # puting the FPS count on the frame 
    cv2.putText(final_output, fps, (7, 70), font, 2, (100, 255, 0), 3, cv2.LINE_AA)
    
    # Get date time from device
    dateran = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    
    # Show the output
    cv2.imshow("Output", final_output)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # If you wanna get picture skeleton uncomment the following four lines
        # dateran = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        # frame = "f_"+ dateran + ".jpg"
        # frame_crop = "c_"+ dateran + ".jpg"
        # cv2.imwrite(frame, datum.cvOutputData)
        # cv2.imwrite(frame_crop, crop_img)
        print("Button pressed")

capture.release()
cv2.destroyAllWindows()