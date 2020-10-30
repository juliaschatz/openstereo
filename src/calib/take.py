import cv2

cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)

cam0.set(cv2.CAP_PROP_FPS, 30)
cam1.set(cv2.CAP_PROP_FPS, 30)

cv2.namedWindow("camL")
cv2.namedWindow("camR")
ct = 0

B = 40
f = 3.6
imgct = 1

while True:
    ret0, frameL = cam0.read()
    ret1, frameR = cam1.read()
    if not (ret0 and ret1):
        print("Camera failed")
        break
    cv2.imshow("camL", frameL)
    cv2.imshow("camR", frameR)

    k = cv2.waitKey(1)
    if k % 256 == 32:
        cv2.imwrite(f"stereo2/left{imgct}.jpg", frameL)
        cv2.imwrite(f"stereo2/right{imgct}.jpg", frameR)
        print(f"Wrote image {imgct}")
        imgct += 1

cam0.release()
cam1.release()
cv2.destroyAllWindows()