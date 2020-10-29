#!/usr/bin/python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_node():
    camR = cv2.VideoCapture(1)
    camL = cv2.VideoCapture(0)

    camR.set(cv2.CAP_PROP_FPS, 30)
    camL.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("disparity")
    #cv2.namedWindow("camL")
    #cv2.namedWindow("camR")
    ct = 0

    B = 40
    f = 3.6
    h = 480
    w = 640
    windowSize = 2
    stereo = cv2.StereoBM_create(numDisparities=144, blockSize=19)
    #cv2.waitKey(0)

    left_dist = np.loadtxt("calib/left_stereo_dist.csv", delimiter=",")
    left_mtx = np.loadtxt("calib/left_stereo_mtx.csv", delimiter=",")
    right_dist = np.loadtxt("calib/right_stereo_dist.csv", delimiter=",")
    right_mtx = np.loadtxt("calib/right_stereo_mtx.csv", delimiter=",")
    R = np.loadtxt("calib/R.csv", delimiter=",")
    T = np.loadtxt("calib/T.csv", delimiter=",")
    Rleft, Rright, Pleft, Pright, Q, validPixROI1, validPixROI2	= cv2.stereoRectify(left_mtx, left_dist, right_mtx, right_dist, (w,h), R,T)
    mapL1, mapL2 =	cv2.initUndistortRectifyMap(left_mtx, left_dist, Rleft, Pleft, (w,h), cv2.CV_8UC1)
    mapR1, mapR2 =	cv2.initUndistortRectifyMap(right_mtx, right_dist, Rright, Pright, (w,h), cv2.CV_8UC1)

    while True:
        ct += 1
        print(ct)
        ret0, frameR = camR.read()
        ret1, frameL = camL.read()
        
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        if not (ret0 and ret1):
            print("Camera failed")
            break
        grayL = cv2.remap(grayL, mapL1, mapL2, cv2.INTER_LINEAR)
        grayR = cv2.remap(grayR, mapR1, mapR2, cv2.INTER_LINEAR)
        disparity = stereo.compute(grayL, grayR)

        _3dimage = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16.0, Q)

        disparity = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow("disparity", disparity)
        cv2.imshow("camL", grayL)
        cv2.imshow("camR", grayR)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            break
        elif k % 256 == 32:
            cv2.imwrite("left.jpg", grayL)
            cv2.imwrite("right.jpg", grayR)

    camR.release()
    camL.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_node()