import cv2
import numpy as np
from coords import *



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
