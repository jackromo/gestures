import freenect, cv2, math
import numpy as np



############## image processing functions ###################



def getDepthMap():
    depth, timestamp = freenect.sync_get_depth()
    # Decrease all values in depth map to within 8 bits to be uint8
    depth = np.clip(depth, 0, 2**10 - 1)
    depth >>= 2
    return depth.astype(np.uint8)


def getMask():
    depth = getDepthMap()
    darkestShade = depth.min()  # darkest shade = closest object (assume is hand)
    ret, thresh = cv2.threshold(depth, darkestShade+20, 255, cv2.THRESH_BINARY_INV)
    return thresh


def getContours(threshImg):
    image, contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours



############### hand processing functions ###################



def getBiggestContour(cnts):
    # m00 moment of a contour = area of enclosed blob
    if len(cnts) == 0: return None
    return max(cnts, key=lambda x: cv2.moments(x)['m00'])


def getBottomLeftPoint(cnt):
    # b-left point of hand not altered by moving fingers of RIGHT hand
    x, y, w, h = cv2.boundingRect(cnt)
    bottomLeftPoint = (x, y+h)
    return bottomLeftPoint


def getBottomRightPoint(cnt):
    # b-right point of hand not altered by moving fingers of LEFT hand
    x, y, w, h = cv2.boundingRect(cnt)
    bottomRightPoint = (x+w, y+h)
    return bottomRightPoint


def findOpenFingerCoords(handCnt):
    hullPnts = getUniqueHullPoints(handCnt)
    if len(hullPnts) < 5: return {}  # if less than 5 fingers, error
    midFingIndex = hullPnts.index(min(hullPnts, key=lambda x: x[1]))  # highest point on hull = middle finger
    # assumes that hand is right hand
    fingers = {'thumb': hullPnts[(midFingIndex-2) % len(hullPnts)],
            'index':    hullPnts[(midFingIndex-1) % len(hullPnts)],
            'middle':   hullPnts[(midFingIndex) % len(hullPnts)],
            'ring':     hullPnts[(midFingIndex+1) % len(hullPnts)],
            'pinky':    hullPnts[(midFingIndex+2) % len(hullPnts)]}  # dict of finger coords
    return fingers


def getUniqueHullPoints(cnt):
    hullPnts = [x[0] for x in cv2.convexHull(cnt).tolist()]
    # point is unique if not too close to any other points (>10px away from other pnts)
    uniquePnts = [hullPnts[i] for i in range(len(hullPnts)) \
            if getDistBetween(hullPnts[i], hullPnts[(i+1) % len(hullPnts)]) > 10]
    return uniquePnts


def getDistBetween(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)



################## main method #######################



def main():
    while True:
        mask = getMask()
        contours = getContours(mask)

        rgbImg = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        handCnt = getBiggestContour(contours)
        img = cv2.drawContours(rgbImg, [handCnt], 0, (0,255,0), 3)
        for c in findOpenFingerCoords(handCnt).values():
            img = cv2.circle(img, tuple(c), 5, (0, 0, 255), -1)
        img = cv2.circle(img, getBottomLeftPoint(handCnt), 5, (255, 0, 0), -1)

        cv2.imshow('image', img)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
