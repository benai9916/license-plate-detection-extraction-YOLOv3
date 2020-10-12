import cv2
import numpy as np
import glob
import random
import os
import gdown


model_exist = os.listdir('model')
if 'yolov3_training_last.weights' not in model_exist:
  url = 'https://drive.google.com/uc?id=1Ez1QHeoIjuHRhQUpgBi-SQU3Qmp4mTtu'
  output = 'model/yolov3_training_last.weights'
  gdown.download(url, output, quiet=False)

else:
  pass

# Load Yolo
net = cv2.dnn.readNet("model/yolov3_training_last.weights",  "model/yolov3_testing.cfg")

# Name custom object
classes = ["License_Plate "]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
def receive_image(images_path):
  # Loading image
  img = cv2.imread(images_path, 1)
  # img = cv2.resize(img, None, fx=0.6, fy=0.6)
  height, width, channels = img.shape

  # Detecting objects
  blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=True)

  net.setInput(blob)
  outs = net.forward(output_layers)

  # Showing informations on the screen
  class_ids = []
  confidences = []
  boxes = []
  predict_score = []
  roi = []
  for out in outs:
      for detection in out:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]
          if confidence > 0.3:
              predict_score.append(confidence)
              # Object detected
              print(class_id)
              center_x = int(detection[0] * width)
              center_y = int(detection[1] * height)
              w = int(detection[2] * width)
              h = int(detection[3] * height)

              # Rectangle coordinates
              x = int(center_x - w / 2)
              y = int(center_y - h / 2)

              boxes.append([x, y, w, h])
              confidences.append(float(confidence))
              class_ids.append(class_id)

  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

  if len(predict_score) != 0:
    final_score = np.round(predict_score[0] * 100)
  else:
    final_score = 0

  font = cv2.FONT_HERSHEY_SIMPLEX
  for i in range(len(boxes)):
      if i in indexes:
          x, y, w, h = boxes[i]
          label = str(classes[class_ids[i]] + str(final_score))
          color = colors[class_ids[i]]
          cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
          cv2.putText(img, label, (x, y - 8), font, 0.7, (0, 255, 0), 2)

          roi = img[y:y+h, x:x+w]

  if len(roi) == 0:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, final_score, img

  else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, final_score, roi