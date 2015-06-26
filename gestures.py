import freenect, cv2, math, time
import numpy as np



############## image processing functions ###################



def getDepthMap():
    depth, timestamp = freenect.sync_get_depth()
    # Decrease all values in depth map to within 8 bits to be uint8
    depth = np.clip(depth, 0, 2**10 - 1)
    depth >>= 2
    depth = cv2.GaussianBlur(depth, (5,5), 0)
    return depth.astype(np.uint8)


def getMask():
    depth = getDepthMap()
    darkestShade = depth.min()  # darkest shade = closest object (assume is hand)
    ret, thresh = cv2.threshold(depth, darkestShade+25, 255, cv2.THRESH_BINARY_INV)
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
    bottomLeftPoint = Point(x, y+h)
    return bottomLeftPoint


def getBottomRightPoint(cnt):
    # b-right point of hand not altered by moving fingers of LEFT hand
    x, y, w, h = cv2.boundingRect(cnt)
    bottomRightPoint = Point(x+w, y+h)
    return bottomRightPoint


def findOpenFingerOffsets(handCnt, isRight=True):
    fingerCoords = findOpenFingerCoords(handCnt, isRight=isRight)
    refPoint = getBottomLeftPoint(handCnt) if isRight else getBottomRightPoint(handCnt)
    fingerOffsets = { k : refPoint.getVectorTo(fingerCoords[k]) for k in fingerCoords.keys()}
    return fingerOffsets


def findOpenFingerCoords(handCnt, isRight=True):
    hullPnts = getUniqueHullPoints(handCnt)
    if len(hullPnts) < 5: return {}  # if less than 5 fingers, error
    midFingIndex = hullPnts.index(min(hullPnts, key=lambda pnt: pnt.getY()))  # highest point on hull = middle finger
    fingerCoords = {}
    fingOffsetsFromMiddle = {'thumb': 2, 'index': 1, 'middle': 0, 'ring': -1, 'pinky': -2}  # offsets for if right hand, take negative val if left hand
    for finger in fingOffsetsFromMiddle.keys():
        offsetFromMid = (1 if isRight else -1) * fingOffsetsFromMiddle[finger]
        fingerCoords[finger] = hullPnts[(midFingIndex + offsetFromMid) % len(hullPnts)]
    return fingerCoords


def anyHullVerticesNear(cnt, point, radius=500):
    hullPnts = getUniqueHullPoints(cnt)
    nearPnts = [p for p in hullPnts if p.getDistTo(point) < radius]
    return len(nearPnts) > 0


def getUniqueHullPoints(cnt):
    if cnt == None: return None
    hullPnts = [Point(pnt[0][0], pnt[0][1]) for pnt in cv2.convexHull(cnt).tolist()]  # pnt = [[x,y]], pnt[0] = [x,y]
    # point is unique if not too close to any other points (>10px away from other pnts)
    uniquePnts = [hullPnts[i] for i in range(len(hullPnts)) \
            if hullPnts[i].getDistTo(hullPnts[(i+1) % len(hullPnts)]) > 10]
    return uniquePnts



################## spatial classes #######################



class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def setX(self, x):
        self.x = x
    def setY(self, y):
        self.y = y
    def toTuple(self):
        return (self.getX(), self.getY())
    def getDistTo(self, p2):
        return math.sqrt((self.getX()-p2.getX())**2 + (self.getY()-p2.getY())**2)
    def getVectorTo(self, p2):
        return Vector(p2.getX()-self.getX(), p2.getY()-self.getY())
    def addToCoord(self, p2):
        return Point(p2.getX()+self.getX(), p2.getY()+self.getY())


class Vector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def setX(self, x):
        self.x = x
    def setY(self, y):
        self.y = y
    def translateCoord(self, pnt):
        return Point(pnt.getX()+self.getX(), pnt.getY()+self.getY())



################# hand classes #######################



class Hand(object):
    def __init__(self, isRight=True):
        self.fingerOffsets = {}
        self.handPos = Point(0,0)
        self.handArea = 0
        self.calibrated = False
        self.isRight = isRight

    def calibrate(self, mask):
        handCnt = self.findHandCnt(mask)
        self.fingerOffsets = findOpenFingerOffsets(handCnt, isRight=self.isRight)
        self.handArea = cv2.moments(handCnt)['m00']
        self.calibrated = True

    def findHandCnt(self, mask):
        contours = getContours(mask)
        return getBiggestContour(contours)

    def getOpenFingers(self, mask):
        handCnt = self.findHandCnt(mask)
        openFingers = {}
        for finger in ['pinky', 'ring', 'middle', 'index', 'thumb']:
            openFingerPos = self.fingerOffsets[finger].translateCoord(self.getHandPos(mask))
            openFingers[finger] = anyHullVerticesNear(handCnt, openFingerPos, radius=40)
        return openFingers

    def getHandPos(self, mask):
        handCnt = self.findHandCnt(mask)
        self.handPos = getBottomLeftPoint(handCnt) if self.isRight else getBottomRightPoint(handCnt)
        return self.handPos



################## main method #######################



def main():  # test method
    hand = Hand(isRight=False)

    while True:
        mask = getMask()
        contours = getContours(mask)

        # draw all close pixels, contours (green), hand's b-left corner (blue), and fingertips (red)

        rgbImg = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        handCnt = getBiggestContour(contours)
        refPoint = getBottomLeftPoint(handCnt) if hand.isRight else getBottomRightPoint(handCnt)
        img = cv2.drawContours(rgbImg, [handCnt], 0, (0,255,0), 3)
        for c in findOpenFingerCoords(handCnt).values():
            img = cv2.circle(img, c.toTuple(), 5, (0, 0, 255), -1)
        img = cv2.circle(img, refPoint.toTuple(), 5, (255, 0, 0), -1)

        # draw checked finger regions if calibrated

        if hand.calibrated:
            fingDict = hand.getOpenFingers(mask)
            for k in hand.fingerOffsets.keys():
                img = cv2.line(img, refPoint.toTuple(), \
                        hand.fingerOffsets[k].translateCoord(refPoint).toTuple(), (255, 0, 255), 3)
                circColor = (0, 255, 255) if fingDict[k] else (0, 0, 127)
                img = cv2.circle(img, hand.fingerOffsets[k].translateCoord(refPoint).toTuple(), 25, circColor, 2)

        cv2.imshow('image', img)

        # request input, if c pressed then calibrate, if g pressed then print open fingers

        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'):   hand.calibrate(mask)
        elif key == ord('g'): print hand.getOpenFingers(mask)


if __name__ == "__main__":
    main()
