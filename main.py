import math, time, cv2 #, freenect
import numpy as np
from hand import *
from coords import *
from depthmap import *
from contours import *
from handstats import *



def main():
    hand = HandStats(isRight=False)

    while True:
        mask = getMask()
        contours = getContours(mask)
        if len(contours)==0: continue

        img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        if hand.isCalibrated() and hand.isOnScreen(mask):
            img = drawHand(img, hand, mask)
        else:
            biggestCnt = getBiggestContour(contours)
            img = highlightCnt(img, biggestCnt)

        cv2.imshow('image', img)
        handleKeyResponse(img, hand, mask)



def drawHand(img, hand, mask):
    fingDict = hand.getOpenFingers(mask)
    if fingDict != None:
        handCnt = hand.findHandCnt(mask)
        refPoint = hand.getHandPos(mask)
        img = highlightCnt(img, handCnt)
        for fing in getFingList():
            img = drawFingDetectionRegion(img, hand, refPoint=refPoint, fingName=fing, isFingOpen=(fingDict[fing]>0.5))
        img = drawHandCntEndFingPoints(img, handCnt)
        img = drawHandRefPoint(img, refPoint)
        img = drawTextInCorner(img, 'Hand Found')
    return img


def highlightCnt(img, cnt):
    img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    return img


def drawHandRefPoint(img, refPoint):
    img = cv2.circle(img, refPoint.toTuple(), 5, (255, 0, 0), -1)
    return img


def drawFingDetectionRegion(img, hand, refPoint=Point(0,0), fingName='thumb', isFingOpen=True):
    img = cv2.line(img, refPoint.toTuple(), \
            hand.fingerOffsets[fingName].translateCoord(refPoint).toTuple(), (255, 0, 255), 3)
    circColor = (0, 255, 255) if isFingOpen else (0, 0, 127)
    img = cv2.circle(img, hand.fingerOffsets[fingName].translateCoord(refPoint).toTuple(), 25, circColor, 2)
    return img


def drawHandCntEndFingPoints(img, handCnt):
    for fingEndCoord in findOpenFingerCoords(handCnt).values():
        img = cv2.circle(img, fingEndCoord.toTuple(), 5, (0, 0, 255), -1)
    return img


def drawTextInCorner(img, text):
    img = cv2.putText(img, text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    return img


def handleKeyResponse(img, hand, mask):
    key = chr(cv2.waitKey(10) & 0xFF)  # if 64 bit system, waitKey() gives result > 8 bits, ANDing with 11111111 removes extra ones

    if key == 'c':   hand.calibrate(mask)
    elif hand.isOnScreen(mask):
        if key == 'g': print hand.getOpenFingers(mask)
        elif key == 'v':
            velocVec = hand.getHandVelocityVec(sampleTimeMsec=100, sampIntervalMsec=10)
            print velocVec.toTuple() if velocVec != None else "no result"
        elif key == 'a':
            accVec = hand.getHandAccelVec(sampleTimeMsec=200, sampIntervalMsec=10)
            print accVec.toTuple() if accVec != None else "no result"




if __name__ == "__main__":
    main()
