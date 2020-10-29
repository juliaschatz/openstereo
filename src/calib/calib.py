import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

camera = sys.argv[1]

subpixsearch = (3,3)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2) * 20

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for i in range(1,60):
    fname = f"{camera}/{i}.jpg"
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    print(i)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img, (9,7),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(img,corners,subpixsearch,(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        #img = cv2.drawChessboardCorners( cv2.cvtColor(img,cv2.COLOR_GRAY2RGB), (9,7), corners2,ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
    else:
        print("Couldn't find corners")
print("starting calib")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (640,480),None,None, flags=(cv2.CALIB_ZERO_TANGENT_DIST))

np.savetxt(f"{camera}_dist.csv", dist, delimiter=",")
np.savetxt(f"{camera}_mtx.csv", mtx, delimiter=",")

errs = []
bad = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    imgnum = i+1
    err = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    errs += [err]
    if err > 0.04:
        bad += [imgnum]
print(f"Bad: {bad}")
print(f"rms: {ret}")
print(f"mean error: {sum(errs)/len(objpoints)}")

cv2.destroyAllWindows()
plt.plot(list(range(1, len(objpoints)+1)), errs)
plt.show()