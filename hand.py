import cv2, math, time
from coords import *
from depthmap import *
from circles import *
from contours import *
import numpy as np



def getOpenFingerVectors(handCnt, palmCirc=None, isRightHand=True):
    if palmCirc==None or handCnt==None: return None
    cent = palmCirc.getCenter()
    fingOffsetsFromMid = getFingIndexOffsetsFromMidFing(isRightHand=isRightHand)
    fingPnts = getOpenFingerPnts(handCnt, palmCirc=palmCirc)
    midFingIndex = getMidFingIndex(fingPnts)
    fingNameLs = getFingList(isRightHand=isRightHand)
    fingPntDict = {fing: fingPnts[(midFingIndex + fingOffsetsFromMid[fing]) % len(fingPnts)] for fing in fingNameLs}
    return {fing: cent.getVectorTo(fingPntDict[fing]) for fing in fingNameLs}


def getOpenFingerPnts(handCnt, palmCirc=None):
    if palmCirc==None or handCnt==None: return None
    fingPnts = []
    poly = getApproxContourPolygon(handCnt, accuracy=0.015)
    cent = palmCirc.getCenter(); rad = palmCirc.getRadius()
    for i in range(len(poly)):
        # fingers at accute angle points, accute if adjacent points are closer to center than it
        pnt = poly[i]; pntAfter = poly[(i+1)%len(poly)]; pntBefore = poly[(i-1)%len(poly)]
        if pntAfter.getDistTo(cent) < pnt.getDistTo(cent) and pntBefore.getDistTo(cent) < pnt.getDistTo(cent):
            if pnt.getDistTo(cent) > (rad+25):
                fingPnts.append(pnt)
    return fingPnts


def getFingAngRegions(handCnt, palmCirc, isRightHand=True):
    fingNames = getFingList(isRightHand=isRightHand)
    fingVecs = getOpenFingerVectors(handCnt, palmCirc, isRightHand=isRightHand)
    betweenFingAngs = getAngsBetweenVecs([fingVecs[fing] for fing in fingNames])
    fingAngBounds = list(reversed(sorted([math.pi]+betweenFingAngs+[0])))  # angs from pi to 0, all angles between finger vectors
    # fingNames in order from leftmost finger's name to right
    # leftmost finger has largest angle to horizontal (<1,0>), so its boundaries = 1rst items in fingAngBounds
    return {fingNames[i]: sorted([fingAngBounds[i], fingAngBounds[i+1]]) for i in range(len(fingNames))}


def getFingIndexOffsetsFromMidFing(isRightHand=True):
    fingers = getFingList(isRightHand=isRightHand)
    fingOffsetsFromMiddle = {fing: fingers.index(fing) - fingers.index('middle') for fing in fingers}
    return fingOffsetsFromMiddle


def getMidFingIndex(hullPnts):
    midFingCoords = min(hullPnts, key=lambda pnt: pnt.getY())  # assume highest point on hull = middle finger
    return hullPnts.index(midFingCoords)


def getAngsBetweenVecs(vecLs):
    vecAngLs = [vec.getAngFromHoriz() for vec in vecLs]
    return [average([vecAngLs[i], vecAngLs[i+1]]) for i in range(len(vecAngLs)-1)]


def getFingList(isRightHand=True):
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    return list(reversed(fingers)) if isRightHand else fingers


def getHighestNotFingPnt(hand, mask):
    handCntPnts = getCntPntLs(hand.findHandCnt(mask))
    noFingerRange = Circle(hand.palmCirc.getCenter(), hand.palmCirc.getRadius()+20)  # make sure highest point for palm isn't fingertip
    cntPntsInErrorBounds = filter(lambda pnt: noFingerRange.containsPnt(pnt), handCntPnts)
    if len(cntPntsInErrorBounds)>0: highestNotFingPnt = min(cntPntsInErrorBounds, key=lambda pnt: pnt.getY())
    else:                           highestNotFingPnt = min(handCntPnts, key=lambda pnt: pnt.getY())
    return highestNotFingPnt



class Hand(object):
    def __init__(self, isRight=True):
        self.fingAngRegions = {}
        self.palmCirc = None
        self.handArea = 0
        self.calibrated = False
        self.isRight = isRight

    def calibrate(self, mask):
        handCnt = getBiggestContour(getContours(mask))
        self.palmCirc = self.getPalmCircle(mask)
        self.fingAngRegions = getFingAngRegions(handCnt, self.palmCirc, isRightHand=self.isRight)
        self.handArea = cv2.moments(handCnt)['m00']
        self.calibrated = True

    def findHandCnt(self, mask):
        contours = getContours(mask)
        # will be None if no viable hand contour found
        if not self.calibrated: return getBiggestContour(contours)
        else: return getContourWithArea(contours, self.handArea, floor=self.handArea/2.5, ceil=self.handArea*2.5)

    def getOpenFingers(self, mask):
        if not self.isOnScreen(mask): return None
        handCnt = self.findHandCnt(mask)
        fingVecs = [self.palmCirc.getCenter().getVectorTo(pnt) for pnt in getOpenFingerPnts(handCnt, self.palmCirc)]
        openFingers = {}
        for finger in getFingList(isRightHand=self.isRight):
            openFingers[finger] = any([self.fingAngRegions[finger][0] <= vec.getAngFromHoriz() <= self.fingAngRegions[finger][1] for vec in fingVecs])
        return openFingers

    def getHandPos(self, mask):
        if not self.isOnScreen(mask): return None
        handCnt = self.findHandCnt(mask)
        self.palmCirc = self.getPalmCircle(mask)
        return self.palmCirc.getCenter()

    def getPalmCircle(self, mask):
        """Assume that hand is open on first run."""
        if self.calibrated and not self.isOnScreen(mask): return None
        handCnt = self.findHandCnt(mask)
        defectPnts = getContourConvexDefects(handCnt, minSize=15, maxSize=80)
        palmCircPnts = []
        if self.palmCirc!=None and (defectPnts==None or len(defectPnts)==0):
            palmCircPnts.append(getHighestNotFingPnt(self, mask))
        else: palmCircPnts += defectPnts
        lowestPnt = max(getCntPntLs(handCnt), key=lambda pnt: pnt.getY())
        palmCircPnts.append(lowestPnt)
        return getSmallestEnclosingCirc(palmCircPnts)

    def isCalibrated(self):
        return self.calibrated

    def isOnScreen(self, mask):
        handCnt = self.findHandCnt(mask)
        return True if handCnt != None else False
