import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

from tool.utils import *
from tool.darknet2onnx import *


def main(namesfile, onnx_file, image_path):

    session = onnxruntime.InferenceSession(onnx_file)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    image_src = cv2.imread(image_path)
    detect(session, image_src, namesfile)



def detect(session, image_src, namesfile):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})

    boxes = post_processing(img_in, 0.4, 0.6, outputs)

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(image_src, boxes[0], savename='predictions_onnx.jpg', class_names=class_names)

    sum = 0
    for i in range(1000):
        prev_time = time.time()
        
        outputs = session.run(None, {input_name: img_in})
        
        curr_time = time.time()
        exec_time = curr_time - prev_time
        if i == 0: continue
        sum += (1 / exec_time)
        info = str(i) + " time:" + str(round(exec_time, 3)) + " average FPS:" + str(round(sum / i, 2)) + ", FPS: " + str(
            round((1 / exec_time), 1))
        print(info)


if __name__ == '__main__':
    print("Inference using onnx ...")
    if len(sys.argv) == 4:
        onnx_file = sys.argv[1]
        namesfile = sys.argv[2]
        image_path = sys.argv[3]
        main(namesfile, onnx_file, image_path)
    else:
        print('Please run this way:\n')
        print('  python demo_onnx.py <weightFile> <namesFile> <imageFile>')
