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



def getContourWithArea(cnts, area, error=200):
    if len(cnts) == 0: return None
    # m00 moment of a contour = area of enclosed blob
    handCnt = min(cnts, key=lambda x: abs(area - cv2.moments(x)['m00']))
    if abs(cv2.moments(handCnt)['m00'] - area) < error: return handCnt
    else: return None


def getBiggestContour(cnts):
    if len(cnts) == 0: return None
    # m00 moment of a contour = area of enclosed blob
    handCnt = max(cnts, key=lambda x: cv2.moments(x)['m00'])
    return handCnt


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


def findOpenFingerOffsets(handCnt):
    fingerCoords = findOpenFingerCoords(handCnt)
    bottomLeft = getBottomLeftPoint(handCnt)
    fingerOffsets = { k : getCoordOffset(bottomLeft, fingerCoords[k]) for k in fingerCoords.keys()}
    return fingerOffsets


def findOpenFingerCoords(handCnt):
    hullPnts = getUniqueHullPoints(handCnt)
    if len(hullPnts) < 5: return {}  # if less than 5 fingers, error
    midFingIndex = hullPnts.index(min(hullPnts, key=lambda x: x[1]))  # highest point on hull = middle finger
    # assumes that hand is right hand
    fingerCoords = {'thumb': hullPnts[(midFingIndex+2) % len(hullPnts)],
            'index':    hullPnts[(midFingIndex+1) % len(hullPnts)],
            'middle':   hullPnts[(midFingIndex) % len(hullPnts)],
            'ring':     hullPnts[(midFingIndex-1) % len(hullPnts)],
            'pinky':    hullPnts[(midFingIndex-2) % len(hullPnts)]}
    return fingerCoords


def anyHullVerticesNear(cnt, point, radius=500):
    hullPnts = getUniqueHullPoints(cnt)
    nearPnts = [p for p in hullPnts if getDistBetween(p, point) < radius]
    return len(nearPnts) > 0


def getUniqueHullPoints(cnt):
    if cnt == None: return None
    hullPnts = [x[0] for x in cv2.convexHull(cnt).tolist()]
    # point is unique if not too close to any other points (>10px away from other pnts)
    uniquePnts = [hullPnts[i] for i in range(len(hullPnts)) \
            if getDistBetween(hullPnts[i], hullPnts[(i+1) % len(hullPnts)]) > 10]
    return uniquePnts


def getDistBetween(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def getCoordOffset(p1, p2):
    return [p2[0]-p1[0], p2[1]-p1[1]]


def addCoords(p1, p2):
    return [p2[0]+p1[0], p2[1]+p1[1]]



################## hand class ########################



class Hand(object):
    def __init__(self):
        self.fingerOffsets = {}
        self.handPos = [0,0]
        self.handArea = 0
        self.calibrated = False

    def calibrate(self, mask):
        contours = getContours(mask)
        handCnt = getBiggestContour(contours)  # assume that biggest contour is hand
        self.fingerOffsets = findOpenFingerOffsets(handCnt)
        self.handArea = cv2.moments(handCnt)['m00']
        self.calibrated = True

    def findHandCnt(self, mask):
        contours = getContours(mask)
        return getBiggestContour(contours)

    def getOpenFingers(self, mask):
        handCnt = self.findHandCnt(mask)
        openFingers = {}
        for finger in ['pinky', 'ring', 'middle', 'index', 'thumb']:
            openFingers[finger] = anyHullVerticesNear(handCnt, addCoords(self.fingerOffsets[finger], self.getHandPos(mask)), radius=25)
        return openFingers

    def getHandPos(self, mask):
        handCnt = self.findHandCnt(mask)
        return getBottomLeftPoint(handCnt)



################## main method #######################



def main():  # test method
    hand = Hand()

    while True:
        mask = getMask()
        contours = getContours(mask)

        # draw all close pixels, contours (green), hand's b-left corner (blue), and fingertips (red)

        rgbImg = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        handCnt = getBiggestContour(contours)
        img = cv2.drawContours(rgbImg, [handCnt], 0, (0,255,0), 3)
        for c in findOpenFingerCoords(handCnt).values():
            img = cv2.circle(img, tuple(c), 5, (0, 0, 255), -1)
        img = cv2.circle(img, getBottomLeftPoint(handCnt), 5, (255, 0, 0), -1)

        # draw checked finger regions if calibrated

        if hand.calibrated:
            for k in hand.fingerOffsets.keys():
                img = cv2.line(img, tuple(getBottomLeftPoint(handCnt)), \
                        tuple(addCoords(hand.fingerOffsets[k], getBottomLeftPoint(handCnt))), (255, 0, 255), 3)
                img = cv2.circle(img, tuple(addCoords(hand.fingerOffsets[k], getBottomLeftPoint(handCnt))), 25, (0, 255, 255), 2)

        cv2.imshow('image', img)

        # request input, if c pressed then calibrate, if g pressed then print open fingers

        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'):   hand.calibrate(mask)
        elif key == ord('g'): print hand.getOpenFingers(mask)


if __name__ == "__main__":
    main()
