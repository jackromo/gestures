from coords import *
import math



class Circle(object):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    def getCenter(self):
        return self.center
    def getRadius(self):
        return self.radius
    def getDiam(self):
        return self.getRadius()*2
    def getCircum(self):
        return self.getDiam()*math.pi
    def getArea(self):
        return math.pi*(self.getRadius()**2)
    def containsPnt(self, pnt):
        return pnt.getDistTo(self.getCenter()) <= self.getRadius()


def getSmallestEnclosingCirc(pntLs):
    print [pnt.toTuple() for pnt in pntLs]
    circle = None
    for (i,pnt) in enumerate(pntLs):
        if circle==None or not circle.containsPnt(pnt):
            circle = getCircOneEdgePoint(pntLs[0:i+1], pnt)
            #print getCircOneEdgePoint(pntLs, pntLs[len(pntLs)-1])
    #print circle
    return circle


def getCircOneEdgePoint(pntLs, p1):
    circle = Circle(p1, 0)
    for (i, p2) in enumerate(pntLs):
        if p2 is p1: continue
        if not circle.containsPnt(p2):
            if circle.getRadius() == 0:
                circle = Circle(Point((p1.getX()+p2.getX())/2, (p1.getY()+p2.getY())/2), p1.getDistTo(p2)/2)
            else:
                circle = getCircTwoEdgePoints(pntLs[0:i+1], p1, p2)
                #print getCircTwoEdgePoints(pntLs[0:i+1], p1, p1)
    return circle


def getCircTwoEdgePoints(pntLs, p1, p2):
    circle = Circle(Point((p1.getX()+p2.getX())/2, (p1.getY()+p2.getY())/2), p1.getDistTo(p2)/2)
    if all([circle.containsPnt(p) for p in pntLs]):
        return circle
    left = None
    right = None
    for p3 in pntLs:
        cross = crossProduct(p1, p2, p3)
        circle = getCircumCircle(p1, p2, p3)
        if circle == None: continue
        elif cross > 0 and (left is None or crossProduct(p1, p2, circle.getCenter()) > crossProduct(p1, p2, left.getCenter())):
            left = circle
        elif cross < 0 and (right is None or crossProduct(p1, p2, circle.getCenter()) < crossProduct(p1, p2, right.getCenter())):
            right = circle

    if right is None or (left is not None and left.getRadius() <= right.getRadius()):
        print "left ", left.getRadius()
        return left
    else:
        print "right", right.getRadius()
        return right


def getCircumCircle(p1, p2, p3):
    ax = p1.getX(); ay = p1.getY()
    bx = p2.getX(); by = p2.getY()
    cx = p3.getX(); cy = p3.getY()
    d = ((ax*(by - cy)) + (bx*(cy - ay)) + (cx*(ay - by))) * 2.0
    if d==0: return None
    x = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    y = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return Circle(Point(x, y), p1.getDistTo(Point(x,y)))


def crossProduct(p0, p1, p2):
    """2*signed area for triangle with corners p0,p1,p2"""
    return (p1.getX() - p0.getX()) * (p2.getY() - p0.getY()) - (p1.getY() - p0.getY()) * (p2.getX() - p0.getX())
