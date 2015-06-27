import math, time, cv2
import numpy as np
from hand import *
from coords import *
from depthmap import *
from contours import *
from handstats import *



# This function is currently spaghetti code, refactoring = priority

def main():
    hand = HandStats(isRight=False)

    while True:
        mask = getMask()
        contours = getContours(mask)

        # draw all close pixels, contours (green), hand's b-left corner (blue), and fingertips (red)

        rgbImg = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        handCnt = getBiggestContour(contours)
        if handCnt == None: continue
        refPoint = getBottomLeftPoint(handCnt) if hand.isRight else getBottomRightPoint(handCnt)
        img = cv2.drawContours(rgbImg, [handCnt], 0, (0,255,0), 3)
        for c in findOpenFingerCoords(handCnt).values():
            img = cv2.circle(img, c.toTuple(), 5, (0, 0, 255), -1)
        img = cv2.circle(img, refPoint.toTuple(), 5, (255, 0, 0), -1)

        # draw checked finger regions if calibrated and hand contour identified

        if hand.isCalibrated() and hand.isOnScreen(mask):
            fingDict = hand.sampleOpenFingersForMsec(msec=50, intervalMsec=10)  # time to sample slows FPS of test
            if fingDict == None: continue
            for k in hand.fingerOffsets.keys():
                img = cv2.line(img, refPoint.toTuple(), \
                        hand.fingerOffsets[k].translateCoord(refPoint).toTuple(), (255, 0, 255), 3)
                circColor = (0, 255, 255) if fingDict[k]>0.5 else (0, 0, 127)
                img = cv2.circle(img, hand.fingerOffsets[k].translateCoord(refPoint).toTuple(), 25, circColor, 2)
            img = cv2.putText(img, 'Hand Found', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow('image', img)

        # request input, if c pressed then calibrate, if g pressed then print open fingers, v and a print veloc and acc

        key = cv2.waitKey(10) & 0xFF  # if 64 bit system, waitKey() gives result > 8 bits, ANDing with 11111111 removes extra ones

        if not hand.isOnScreen(mask): continue

        if key == ord('c'):   hand.calibrate(mask)
        elif key == ord('g'): print hand.getOpenFingers(mask)
        elif key == ord('v'): print hand.getHandVelocityVec(sampleTimeMsec=50, sampIntervalMsec=10).toTuple()
        elif key == ord('a'): print hand.getHandAccelVec(sampleTimeMsec=100, sampIntervalMsec=10).toTuple()


if __name__ == "__main__":
    main()
