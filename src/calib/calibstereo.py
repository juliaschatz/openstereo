import numpy as np
import cv2
import sys

show = sys.argv[1] == "show"

subpixsearch = (3,3)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2) * 20 # mm size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
leftimgpoints = [] # 2d points in image plane.
rightimgpoints = [] # 2d points in image plane.

left_dist = np.loadtxt("left_dist.csv", delimiter=",")
left_mtx = np.loadtxt("left_mtx.csv", delimiter=",")
right_dist = np.loadtxt("right_dist.csv", delimiter=",")
right_mtx = np.loadtxt("right_mtx.csv", delimiter=",")

for i in range(1, 40):
    imgleft = cv2.imread(f"stereo2/left{i}.jpg", cv2.IMREAD_GRAYSCALE)
    imgright = cv2.imread(f"stereo2/right{i}.jpg", cv2.IMREAD_GRAYSCALE)
    if imgleft is None or imgright is None:
        continue
    print(i)

    # Find the chess board corners
    retl, lcorners = cv2.findChessboardCorners(imgleft, (9,7),None)
    retr, rcorners = cv2.findChessboardCorners(imgright, (9,7),None)

    # If found, add object points, image points (after refining them)
    if retl and retr:
        objpoints.append(objp)

        lcorners2 = cv2.cornerSubPix(imgleft,lcorners,subpixsearch,(-1,-1),criteria)
        leftud = sum(lcorners2[-1] - lcorners2[0]) < 0
        rcorners2 = cv2.cornerSubPix(imgright,rcorners,subpixsearch,(-1,-1),criteria)
        rightud = sum(rcorners2[-1] - rcorners2[0]) < 0
        if leftud.any():
            lcorners2 = np.flip(lcorners2, 0)
        if rightud.any():
            rcorners2 = np.flip(rcorners2, 0)
        leftimgpoints.append(lcorners2)
        rightimgpoints.append(rcorners2)

        # Draw and display the corners
        if show:
            imgleft = cv2.drawChessboardCorners(cv2.cvtColor(imgleft,cv2.COLOR_GRAY2RGB), (9,7), lcorners2,retl)
            cv2.imshow('imgleft',imgleft)
            imgright = cv2.drawChessboardCorners(cv2.cvtColor(imgright,cv2.COLOR_GRAY2RGB), (9,7), rcorners2,retr)
            cv2.imshow('imgright',imgright)
            cv2.waitKey(00)
    else:
        print("Couldn't find corners")
print("starting calib")
retval, left_mtx, left_dist, right_mtx, right_dist, R, T, E, F =	cv2.stereoCalibrate(objpoints, leftimgpoints, rightimgpoints, 
                                                                                   left_mtx, left_dist, right_mtx, right_dist, (640,480), criteria=criteria,
                                                                                   flags=(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH))
print(retval)
np.savetxt(f"left_stereo_dist.csv", left_dist, delimiter=",")
np.savetxt(f"left_stereo_mtx.csv", left_mtx, delimiter=",")
np.savetxt(f"right_stereo_dist.csv", right_dist, delimiter=",")
np.savetxt(f"right_stereo_mtx.csv", right_mtx, delimiter=",")
np.savetxt(f"R.csv", R, delimiter=",")
np.savetxt(f"T.csv", T, delimiter=",")

cv2.destroyAllWindows()