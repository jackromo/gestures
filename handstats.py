import math, time, cv2
import numpy as np
from depthmap import *
from hand import *
from coords import *



def performFuncOverInterval(func, *args, **kwargs):
    interval = kwargs.pop('interval', 0.01)  # only kwarg = interval, default val = 0.01
    intervalStartTime = time.time()
    result = func(*args)
    timeTakenToSample = time.time() - intervalStartTime
    sleepTime = interval-timeTakenToSample if timeTakenToSample < interval else 0
    time.sleep(sleepTime)
    return result


class HandStats(Hand):
    """child class of Hand with sampling capabilities"""

    def sampleOpenFingersForMsec(self, msec=50, intervalMsec=10):
        """Return fraction of time, 0<=t<=1, that each finger is open during sampling time."""
        openFingerSamples={fing: [] for fing in getFingList()}
        startTime = time.time()
        while (time.time() - startTime) < (msec/1000.0):
            openFingerSamples = performFuncOverInterval(self.addNewFingPosSample, openFingerSamples, interval=intervalMsec/1000.0)
        if any([len(fingSampLs) == 0 for fingSampLs in openFingerSamples.values()]): return None
        openFingerAverages = {fing: average(openFingerSamples[fing]) for fing in getFingList()}
        return openFingerAverages

    def addNewFingPosSample(self, prevSamps):
        fingersSample = self.getOpenFingers(getMask())
        if fingersSample != None:
            for fing in getFingList(): prevSamps[fing].append(fingersSample[fing])
        return prevSamps

    def sampleHandPosForMsec(self, msec=50, intervalMsec=10):
        handPosList = []
        startTime = time.time()
        while (time.time() - startTime) < (msec/1000.0):
            handPosList = performFuncOverInterval(self.addNewHandPosSample, handPosList, interval=intervalMsec/1000.0)
        return handPosList

    def addNewHandPosSample(self, prevSamps):
        handPos = self.getHandPos(getMask())
        if handPos != None: prevSamps.append(handPos)
        return prevSamps

    def getHandVelocityVec(self, sampleTimeMsec=50, sampIntervalMsec=10):
        handPosList = self.sampleHandPosForMsec(msec=sampleTimeMsec, intervalMsec=sampIntervalMsec)
        if len(handPosList)<2: return None  # if less than two positions, cannot calculate speed
        getSpeedVec = lambda p1, p2, time: Vector((p2.getX()-p1.getX())/time, (p2.getY()-p1.getY())/time)
        speedsList = [getSpeedVec(handPosList[i], handPosList[i+1], sampIntervalMsec/1000.0) for i in range(len(handPosList)-1)]
        averageSpeedVec = Vector(average([p.getX() for p in speedsList]), average([p.getY() for p in speedsList]))
        return averageSpeedVec

    def getHandAccelVec(self, sampleTimeMsec=50, sampIntervalMsec=10):
        """assume acceleration is constant during sampling"""
        speedSampleTime = sampleTimeMsec/2.0
        veloc1 = self.getHandVelocityVec(sampleTimeMsec=speedSampleTime, sampIntervalMsec=sampIntervalMsec)
        veloc2 = self.getHandVelocityVec(sampleTimeMsec=speedSampleTime, sampIntervalMsec=sampIntervalMsec)
        if veloc1==None or veloc2==None: return None
        accelerationVec = Vector((veloc2.getX()-veloc1.getX())/(sampleTimeMsec/1000.0), (veloc2.getY()-veloc1.getY())/(sampleTimeMsec/1000.0))
        return accelerationVec
