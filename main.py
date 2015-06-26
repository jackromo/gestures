import math, time, cv2
import numpy as np
from hand import *
from coords import *
from depthmap import *
from contours import *



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
