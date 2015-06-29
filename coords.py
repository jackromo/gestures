import math



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
        return (self.getX(), self.getY())
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
    def multWithConst(self, const):
        return Vector(self.getX()*const, self.getY()*const)
    def translateCoord(self, pnt):
        return Point(pnt.getX()+self.getX(), pnt.getY()+self.getY())
    def toTuple(self):
        return (self.getX(), self.getY())
