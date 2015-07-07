import math, time, cv2
import numpy as np
from hand import *
from coords import *
from depthmap import *
from contours import *
from handstats import *
from circles import *



def main():
    hand = HandStats(isRight=True)

    while True:
        mask = getMask()
        depthMap = getDepthMap()
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
        handCirc = hand.getPalmCircle(mask)
        img = highlightCnt(img, handCnt)
        img = drawCntPolygon(img, handCnt)
        img = drawFingPoints(img, hand, mask, openFingLs=fingDict)
        img = drawPalmCircle(img, handCirc)
        img = drawTextInCorner(img, 'Hand found')
    return img


def drawFingPoints(img, hand, mask, openFingLs=None):
    fingPnts = getOpenFingerPnts(hand.findHandCnt(mask), palmCirc=hand.getPalmCircle(mask))
    for pnt in fingPnts: img = cv2.circle(img, pnt.toTuple(), 10, (127,127,127), 2)
    return img


def drawCntPolygon(img, cnt):
    poly = getApproxContourPolygon(cnt, accuracy=0.01)
    for i in range(len(poly)):
        img = cv2.line(img, poly[i].toTuple(), poly[(i+1)%len(poly)].toTuple(), (0, 0, 255), 3)
    return img


def highlightCnt(img, cnt):
    img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    return img


def drawPalmCircle(img, handCirc):
    img = cv2.circle(img, handCirc.getCenter().toTuple(), int(handCirc.getRadius()), (255, 0, 0), 3)
    img = cv2.circle(img, handCirc.getCenter().toTuple(), 5, (255, 0, 0), -1)
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
            velocVec = hand.getHandVelocityVec(sampleTimeMsec=150, sampIntervalMsec=10)
            print velocVec.toTuple() if velocVec != None else "no result"
        elif key == 'a':
            accVec = hand.getHandAccelVec(sampleTimeMsec=200, sampIntervalMsec=10)
            print accVec.toTuple() if accVec != None else "no result"



if __name__ == "__main__":
    main()
