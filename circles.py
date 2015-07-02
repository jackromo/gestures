from coords import *
import math



class Circle(object):
    def __init__(self, center, radius):
        self.center = Point(float(center.getX()), float(center.getY()))
        self.radius = float(radius)
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
        return pnt.getDistTo(self.getCenter()) <= (self.getRadius()+1) # margin of error = 1


def getSmallestEnclosingCirc(pntLs):
    if pntLs==None or len(pntLs)==0: return None
    elif len(pntLs)==1: return Circle(pntLs[0], 0)
    candidateCircs = getAllPossibleEnclosingCircs(pntLs)
    if len(candidateCircs)==0: return None
    return min(candidateCircs, key=lambda circ: circ.getArea())


def getAllPossibleEnclosingCircs(pntLs):
    pntPairs = getAllUniqueItemPairs(pntLs)
    pntTrips = getAllUniqueItemTriplets(pntLs)
    twoPntCircs = [getCircTwoPointsOnDiam(pair[0],pair[1]) for pair in pntPairs]
    threePntCircs = [getCircumCircle(trip[0], trip[1], trip[2]) for trip in pntTrips]
    allPossibleEnclosingCircs = twoPntCircs + threePntCircs
    return filter(lambda circ: circ is not None and all([circ.containsPnt(pnt) for pnt in pntLs]), allPossibleEnclosingCircs)


def getCircumCircle(p1, p2, p3):
    """algorithm from Wikipedia (see circumcircle: cartesian coordinates)"""
    ax = p1.getX(); ay = p1.getY()
    bx = p2.getX(); by = p2.getY()
    cx = p3.getX(); cy = p3.getY()
    d = ((ax*(by - cy)) + (bx*(cy - ay)) + (cx*(ay - by))) * 2.0
    if d==0: return None
    x = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    y = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    center = Point(x, y)
    return Circle(center, p1.getDistTo(center))


def getCircTwoPointsOnDiam(p1, p2):
    midPoint = Point(average([p1.getX(), p2.getX()]), average([p1.getY(), p2.getY()]))
    radius = p1.getDistTo(p2)/2
    return Circle(midPoint, radius)


def getAllUniqueItemPairs(ls):
    return [(ls[i], ls[j]) for j in range(len(ls)-1) for i in range(j+1, len(ls))]


def getAllUniqueItemTriplets(ls):
    return [(ls[j], ls[i], ls[k]) for j in range(len(ls)-2) for i in range(j+1, len(ls)-1) for k in range(i+1, len(ls))]
