import sys
sys.path.append("")
import os 
import numpy as np
import imutils
import cv2

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = "models/face_detector/deploy.prototxt"
modelPath = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")

TF_MODEL_FILE_PATH = 'models/liveness/efficientnet_model.tflite' # The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
print(interpreter.get_signature_list())

classify_lite = interpreter.get_signature_runner('serving_default')
print("classify_lite", classify_lite)

img_height = 200
img_width = 200

cap = cv2.VideoCapture("test_images/fake3.mp4")
cap.set(3,640)
cap.set(4,480)


while(True):
    # break
    ret, frame = cap.read()

	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
    frame = imutils.resize(frame, height=480, width=640)

	# grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the detected bounding box does fall outside the
            # dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extract the face ROI and then preproces it in the exact
            # same manner as our training data
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (img_height, img_width))
            # Convert the resized face image to a format compatible with TensorFlow
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = tf.expand_dims(face, 0)  # Create a batch


            predictions_lite = classify_lite(input_2=face)['fc2']
            score_lite = predictions_lite[0]
            print("score_lite", score_lite)
            
            if score_lite[0] > 0.8:
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
                _label = "Liveness: {:.4f}".format(score_lite[0])
                cv2.putText(frame, _label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
                _label = "Fake: {:.4f}".format(1-score_lite[0])
                cv2.putText(frame, _label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    cv2.imshow('frame', frame)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()