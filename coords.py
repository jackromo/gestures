import math


def average(ls):
    if len(ls)>0: return sum(ls)/float(len(ls))
    else: return 0


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def setX(self, x):
        self.x = x
    def setY(self, y):
        self.y = y
    def toTuple(self):
        return (int(self.getX()), int(self.getY()))
    def getDistTo(self, p2):
        return math.sqrt((self.getX()-p2.getX())**2 + (self.getY()-p2.getY())**2)
    def getVectorTo(self, p2):
        return Vector(p2.getX()-self.getX(), p2.getY()-self.getY())
    def addToCoord(self, p2):
        return Point(p2.getX()+self.getX(), p2.getY()+self.getY())


class Vector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def setX(self, x):
        self.x = x
    def setY(self, y):
        self.y = y
    def getGradient(self):
        return self.y/float(self.x)
    def getLength(self):
        return Point(0, 0).getDistTo(Point(self.getX(), self.getY()))
    def multWithConst(self, const):
        return Vector(self.getX()*const, self.getY()*const)
    def dotProdWith(self, vec):
        return (self.getX()*vec.getX()) + (self.getY()*vec.getY())
    def translateCoord(self, pnt):
        return Point(pnt.getX()+self.getX(), pnt.getY()+self.getY())
    def getAngFromHoriz(self):
        #angle in radians
        if self.getX()!=0:
            dot = self.dotProdWith(Vector(1, 0))
            return math.acos(dot/float(self.getLength()))
        elif self.getY()>0: return math.pi/2.0
        elif self.getY()<0: return -math.pi/2.0
        else: return None
    def toTuple(self):
        return (int(self.getX()), int(self.getY()))
