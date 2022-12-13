import cv2 as cv
import time
import numpy as np
from urllib.request import urlopen
import requests
import os
import tensorflow as tf
import pyrebase

config = {"apiKey": "AIzaSyAoRu5jQR0__ioiQIRSwLj01bt9fazDt14",
  "authDomain": "trashdetection-67763.firebaseapp.com",
  "databaseURL" : "trashdetection-67763.firebaseio.com/",
  "projectId": "trashdetection-67763",
  "storageBucket": "trashdetection-67763.appspot.com",
  "messagingSenderId": "31671801265",
  "appId": "1:31671801265:web:c8f6d9620a7c755ce1d990",
  "measurementId": "G-D9X8393TDD"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

input_size = (224, 224)

url = "http://192.168.40.31:8080/video"
cam = cv.VideoCapture(url)

cam.set(cv.CAP_PROP_BUFFERSIZE, 5)

model = tf.keras.models.load_model("keras_model.h5", compile = False)

isTake = 3

oldURL = ""

while True:
    start_time = time.time()
    
    
    ret, frame = cam.read()

    # Resize the frame for the model
    model_frame = cv.resize(frame, input_size, frame)
    # Expand Dimension (224, 224, 3) -> (1, 224, 224, 3) and Normalize the data
    model_frame = np.expand_dims(model_frame, axis=0) / 255.0

    # Predict
    is_mask_prob = model.predict(model_frame)[0]
    is_mask = np.argmax(is_mask_prob)

    # Compute the model inference time
    inference_time = time.time() - start_time
    fps = 1 / inference_time
    fps_msg = "Time: {:05.1f}ms {:.1f} FPS".format(inference_time * 1000, fps)
    
    
    
 
    # Add Information on screen
    if is_mask == 0:
        msg_mask = "Human"
        URL = "http://192.168.40.92:8080/nomal"
    elif is_mask == 1:
        msg_mask = "!!!"
        URL = "http://192.168.40.92:8080/high"
        if(isTake > 0):
            now = time.strftime('%Y%m%d_%H%M%S.png')
            cv.imwrite(now, frame)
            storage.child("cctv_Image/" + now).put(now)
            os.remove(now)
            isTake -= 1
        #time.sleep(5)
        
    else:
        msg_mask = "no one detected"
        URL = "http://192.168.40    .92:8080"
        isTake = 3

    if oldURL != URL:
        try:
            oldURL = URL
            response = requests.get(URL)
        except:
            print("")
        
    msg_mask += " ({:.01f})%".format(is_mask_prob[is_mask] * 100)

   
    cv.putText(frame, msg_mask, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    
    cv.imshow("stream", frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break
cam.release()
cv.destroyAllWindows()