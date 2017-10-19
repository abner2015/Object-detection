#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import cv2 as cv3
import time
#from imutils.video import FPS
CLASSES = ('__background__',
           '1', '2', '3', '4',
           '5', '6', '7', '8', '9',
           '10', '11', '12', '13',
           '14', '15', '16',
           '17', '18', '19', '20')

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def detect_vedio():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #prototxt = 'py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/police.prototxt'
    prototxt = 'deploy.prototxt'
    caffemodel = 'train.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    vs = cv3.VideoCapture("/home/abner/Desktop/10163.ts")
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #print ('aaa', cap.isOpened())
    size = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    write = cv2.VideoWriter('/home/abner/Desktop/result5.avi', fourcc, 29, size)
    #time.sleep(1.0)
    #fps = FPS().start()
    print('open ', vs.isOpened())
    # loop over the frames from the video stream
    while True:

        # grap the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels

        ret, frame = vs.read()
        # print('shape : ', type(frame))
        if frame is None:
            break

        (h, w) = frame.shape[:2]
        print('h : ', h)
        print('w : ', w)
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, frame)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class
        CONF_THRESH = 0.6
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            #detections = dets[keep, :]
            dets = dets[keep, :]
            class_name = cls
            thresh = CONF_THRESH
            inds = np.where(dets[:, -1] >= thresh)[0]
            if len(inds) == 0:
                continue

            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                x1 = int(bbox[2] - w)
                y1 = int(bbox[3])
                x2 = int(bbox[0] + w)
                y2 = int(bbox[1])
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                          COLORS[1], 2)
                y = bbox[2] - 15 if bbox[2] - 15 > 15 else bbox[2] + 15
                cv3.putText(frame, class_name, (int(bbox[0]), int(bbox[1])),
                        cv3.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        write.write(frame)
        cv3.imshow("Frame", frame)

        key = cv3.waitKey(1) & 0xff
        if key == ord('q'):
            break
        #fps.update()

    #fps.stop()

    vs.release()
    write.release()
    cv3.destroyAllWindows()
if __name__ == '__main__':
    detect_vedio()
