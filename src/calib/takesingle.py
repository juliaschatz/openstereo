import cv2
import sys

side = sys.argv[1]

cam = cv2.VideoCapture({"left": 0, "right": 1}[side])

cam.set(cv2.CAP_PROP_FPS, 30)

cv2.namedWindow("cam")
ct = 0

B = 40
f = 3.6
imgct = 1

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        print("Camera failed")
        break
    cv2.imshow("cam", gray)

    k = cv2.waitKey(1)
    if k % 256 == 32:
        cv2.imwrite(f"{side}/{imgct}.jpg", gray)
        print(f"Wrote image {imgct}")
        imgct += 1

cam.release()
cv2.destroyAllWindows()