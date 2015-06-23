import freenect
import cv2
import numpy as np



def getDepthMap():
    depth, timestamp = freenect.sync_get_depth()
    # Decrease all values in depth map to within 8 bits to be uint8
    depth = np.clip(depth, 0, 2**10 - 1)
    depth >>= 2
    return depth.astype(np.uint8)


def getMask():
    depth = getDepthMap()
    ret, thresh = cv2.threshold(depth, 160, 255, cv2.THRESH_BINARY_INV)
    return thresh


def getContours(threshImg):
    image, contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def getBiggestContour(cnts):
    # m00 moment of a contour = area of enclosed blob
    if len(cnts) == 0: return None
    return max(cnts, key=lambda x: cv2.moments(x)['m00'])



################## main method #######################



def main():
    while True:
        mask = getMask()
        contours = getContours(mask)

        rgbImg = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        img = cv2.drawContours(rgbImg, [getBiggestContour(contours)], 0, (0,255,0), 3)

        cv2.imshow('image', img)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
