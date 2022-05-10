
import streamlit as st
import time
import numpy as np
import plotly.express as px
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from base64 import b64decode, b64encode
import numpy as np
import PIL
import io
import html
import time
import matplotlib.pyplot as plt
import os
import cv2 as cv
import time

col1, col2 = st.columns([1,3])

with col1:

  Conf_threshold = st.slider("This is used for confidence threshold to know how much confidence are we in detecting", min_value=0, max_value=1, step=0.1)

  NMS_threshold = st.slider("This will tell us how much boxes we want greater the number more the boxes", min_value=0, max_value=1, step=0.1)

with col2:

  st.title("OBJECT DETECTION")

  st.header("Currently it is available for videos only whether uploaded or demo")


  COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)]

  class_name = ["debris"]

  net = cv.dnn.readNet('yolov4-custom_best.weights', 'yolov4-custom.cfg')

  net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
  net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

  model = cv.dnn_DetectionModel(net)
  model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

  cap = st.file_uploader("Upload the video Debris file you want to detect", type=['jpg'])

  if cap is None :

    st.write("As any video is not uploaded so Using the demo video")

    cap = cv.VideoCapture('part12.mp4')

  starting_time = time.time()

  frame_counter = 0

  while True:

      ret, frame = cap.read()

      frame_counter += 1

      if ret == False:
          break

      classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

      for (classid, score, box) in zip(classes, scores, boxes):

          color = COLORS[int(classid) % len(COLORS)]
          label = "%s : %f" % (class_name[classid[0]], score)
          cv.rectangle(frame, box, color, 1)
          cv.putText(frame, label, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
          
      endingTime = time.time() - starting_time

      fps = frame_counter/endingTime

      cv.putText(frame, f'FPS: {fps}', (20, 50),
                cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
      
      st.video(frame)
