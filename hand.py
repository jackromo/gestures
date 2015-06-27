import cv2, math, time
from coords import *
from depthmap import *
from contours import *
import numpy as np



def average(ls):
    return sum(ls)/float(len(ls))


def findOpenFingerOffsets(handCnt, isRightHand=True):
    if handCnt==None: return None
    fingerCoords = findOpenFingerCoords(handCnt, isRightHand=isRightHand)
    refPoint = getBottomLeftPoint(handCnt) if isRightHand else getBottomRightPoint(handCnt)
    fingerOffsets = { k : refPoint.getVectorTo(fingerCoords[k]) for k in fingerCoords.keys()}
    return fingerOffsets


def findOpenFingerCoords(handCnt, isRightHand=True):
    """Get middle finger index. Vertex of hull before it is index finger, after is ring, 2 before is thumb, etc."""
    if handCnt==None: return None
    hullPnts = getUniqueHullPoints(handCnt)
    if len(hullPnts) < 5: return {}  # if less than 5 fingers, error
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
        self.handPos = Point(0,0)
        self.handArea = 0
        self.calibrated = False
        self.isRight = isRight

    def calibrate(self, mask):
        handCnt = getBiggestContour(getContours(mask))
        self.fingerOffsets = findOpenFingerOffsets(handCnt, isRightHand=self.isRight)
        self.handArea = cv2.moments(handCnt)['m00']
        self.calibrated = True

    def findHandCnt(self, mask):
        contours = getContours(mask)
        # will be None if no viable hand contour found
        return getContourWithArea(contours, self.handArea, floor=self.handArea/2.5, ceil=self.handArea+3000)

    def getOpenFingers(self, mask):
        if not self.isOnScreen(mask): return None
        handCnt = self.findHandCnt(mask)
        openFingers = {}
        for finger in getFingList(isRightHand=self.isRight):
            openFingerPos = self.fingerOffsets[finger].translateCoord(self.getHandPos(mask))
            openFingers[finger] = anyHullVerticesNear(handCnt, openFingerPos, radius=40)
        return openFingers

    def getHandPos(self, mask):
        if not self.isOnScreen(mask): return None
        handCnt = self.findHandCnt(mask)
        self.handPos = getBottomLeftPoint(handCnt) if self.isRight else getBottomRightPoint(handCnt)
        return self.handPos

    def isCalibrated(self):
        return self.calibrated

    def isOnScreen(self, mask):
        handCnt = self.findHandCnt(mask)
        return True if handCnt != None else False
