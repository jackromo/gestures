import cv2, math, time
from coords import *
from depthmap import *
from circles import *
from contours import *
import numpy as np



def findOpenFingerOffsets(handCnt, refPoint, isRightHand=True):
    if handCnt==None: return None
    fingerCoords = findOpenFingerCoords(handCnt, isRightHand=isRightHand)
    fingerOffsets = { k : refPoint.getVectorTo(fingerCoords[k]) for k in fingerCoords.keys()}
    return fingerOffsets


def findOpenFingerCoords(handCnt, isRightHand=True):
    """Get middle finger index. Vertex of hull before it is index finger, after is ring, 2 before is thumb, etc."""
    if handCnt==None: return None
    hullPnts = getUniqueHullPoints(handCnt)
    midFingIndex = getMidFingIndex(hullPnts)
    fingOffsetsFromMid = getFingIndexOffsetsFromMidFing(isRightHand=isRightHand)
    fingerCoords = {fing : hullPnts[(midFingIndex + fingOffsetsFromMid[fing]) % len(hullPnts)] for fing in getFingList(isRightHand=isRightHand)}
    return fingerCoords


def getFingIndexOffsetsFromMidFing(isRightHand=True):
    fingers = getFingList(isRightHand=isRightHand)
    fingOffsetsFromMiddle = {fing: fingers.index(fing) - fingers.index('middle') for fing in fingers}
    return fingOffsetsFromMiddle


def getMidFingIndex(hullPnts):
    midFingCoords = min(hullPnts, key=lambda pnt: pnt.getY())  # assume highest point on hull = middle finger
    return hullPnts.index(midFingCoords)


def getFingList(isRightHand=True):
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    return list(reversed(fingers)) if isRightHand else fingers



class Hand(object):
    def __init__(self, isRight=True):
        self.fingerOffsets = {}
        self.palmCirc = None
        self.handArea = 0
        self.calibrated = False
        self.isRight = isRight

    def calibrate(self, mask):
        handCnt = getBiggestContour(getContours(mask))
        self.palmCirc = self.getPalmCircle(mask)
        refPoint = self.palmCirc.getCenter()
        self.fingerOffsets = findOpenFingerOffsets(handCnt, refPoint, isRightHand=self.isRight)
        self.handArea = cv2.moments(handCnt)['m00']
        self.calibrated = True

    def findHandCnt(self, mask):
        contours = getContours(mask)
        # will be None if no viable hand contour found
        if not self.calibrated: return getBiggestContour(contours)
        else: return getContourWithArea(contours, self.handArea, floor=self.handArea/2.5, ceil=self.handArea+3000)

    def getOpenFingers(self, mask):
        if not self.isOnScreen(mask): return None
        handCnt = self.findHandCnt(mask)
        openFingers = {}
        for finger in getFingList(isRightHand=self.isRight):
            fingPosIfOpen = self.fingerOffsets[finger].translateCoord(self.getHandPos(mask))
            openFingers[finger] = True if anyHullVerticesNear(handCnt, fingPosIfOpen, radius=25) else False
        return openFingers

    def getHandPos(self, mask):
        if not self.isOnScreen(mask): return None
        handCnt = self.findHandCnt(mask)
        self.palmCirc = self.getPalmCircle(mask)
        return self.palmCirc.getCenter()

    def getPalmCircle(self, mask):
        """Assume that hand is open on first run."""
        # TOO LONG, REFACTOR LATER
        if self.calibrated and not self.isOnScreen(mask): return None
        handCnt = self.findHandCnt(mask)
        defectPnts = getContourConvexDefects(handCnt, minSize=15, maxSize=80)
        cntPnts = [Point(pnt[0][0], pnt[0][1]) for pnt in handCnt.tolist()]
        palmCircPnts = []
        if self.palmCirc != None:
            noFingerRange = Circle(self.palmCirc.getCenter(), self.palmCirc.getRadius()+20)  # make sure highest point for palm isn't fingertip
            cntPntsInErrorBounds = filter(lambda pnt: noFingerRange.containsPnt(pnt), cntPnts)
            if len(cntPntsInErrorBounds)>0: highestNotFingPnt = min(cntPntsInErrorBounds, key=lambda pnt: pnt.getY())
            else:                           highestNotFingPnt = min(cntPnts, key=lambda pnt: pnt.getY())
        if defectPnts==None or len(defectPnts)==0: palmCircPnts.append(highestNotFingPnt)
        else: palmCircPnts += defectPnts
        lowestPnt = max(cntPnts, key=lambda pnt: pnt.getY())
        palmCircPnts.append(lowestPnt)
        return getSmallestEnclosingCirc(palmCircPnts)

    def isCalibrated(self):
        return self.calibrated

    def isOnScreen(self, mask):
        handCnt = self.findHandCnt(mask)
        return True if handCnt != None else False
