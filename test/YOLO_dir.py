import time
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--direction', required=True,
                help='path to input image dir')
ap.add_argument('-s', '--save', required=True,
                help='path to output detection txt')    
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

count_time = 0
count = 0

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, "{:.2f}% {}".format(confidence * 100, label), (x + 15, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

classes = None
count_object = [0, 0, 0, 0, 0]

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

for filename in glob.glob("{}/*.jpg".format(args.direction)):
    count = count + 1
    print("open {}".format(filename))
    image = cv2.imread(filename)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    txt_file = args.save + "/" + filename.split("\\")[-1].split("/")[-1].split(".")[0] + ".txt"
    print("Save detected: {} file".format(txt_file))

    txt_file = open(txt_file, "w+")

    # Thực hiện xác định bằng HOG và SVM
    start = time.time()

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                txt_file.write("{} {} {} {} {}\n".format(class_id, detection[0], detection[1], detection[2], detection[3]))
                count_object[class_id] = count_object[class_id] + 1
    end = time.time()
    txt_file.close()

    print("YOLO Execution time: " + str(end-start))
    count_time = count_time + end - start
	
print("Detect: {}".format(count_object))
print("Time detection: {}".format(count_time))
print("Average time {}".format(count_time / count))
print("FPS: {}".format(1.0 / (count_time / float(count))))