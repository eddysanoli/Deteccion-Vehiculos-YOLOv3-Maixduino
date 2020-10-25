import math
import time

class deque:
    def __init__(self, maxLen):
        self.list = []
        self.maxLen = maxLen
        self.currentLen = len(self.list)

    def insert(self, value):
        if len(self.list) < self.maxLen:
            tempList = [0.0 for i in range(0,len(self.list)+1)]
            tempList[0] = value
            for i in range(len(self.list)):
                tempList[i+1] = self.list[i]
            self.list = tempList
        else:
            for i in range(1,len(self.list)):
                self.list[-i] = self.list[-i-1]
            self.list[0] = value

    def avg(self):
        return sum(self.list)/len(self.list)

    def isNotEmpty(self):
        return len(self.list) > 0

    def lastElement(self):
        return self.list[0]

    def clean(self):
        self.list = []


class path:
    def __init__(self, maxLen, maxDist):
        self.path = {'x': deque(maxLen), 'y': deque(maxLen)}
        self.maxDist = maxDist
        self.maxLen = maxLen
        self.timeRef = time.time()

    def feedWatchDog(self):
        self.timeRef = time.time()

    def watchDog(self):
        interval = time.time() - self.timeRef
        if interval > 200:
            self.timeRef = time.time()
            self.clean()

    def getPath(self):
        return {'x': self.path['x'].list, 'y': self.path['y'].list}

    def clean(self):
        self.path['x'].clean()
        self.path['y'].clean()

    def isNotEmpty(self):
        return self.path['x'].isNotEmpty() and self.path['y'].isNotEmpty()

    def getLen(self):
        if(len(self.path['x'].list) == len(self.path['y'].list)):
            return len(self.path['x'].list)
        else:
            return 0

    def lastPoint(self):
        return [self.path['x'].list[0], self.path['y'].list[0]]

    def indexedPoint(self, index):
        return [self.path['x'].list[index], self.path['y'].list[index]]

    def distFromLast(self, point = [0,0]):
        dist = math.sqrt((point[0] - self.lastPoint()[0])**2 +(point[1] - self.lastPoint()[1])**2)
        return dist

    def insertPoint(self, point = [0,0]):
        if(self.isNotEmpty()):
            if(self.distFromLast(point) < self.maxDist):
                self.path['x'].insert(point[0])
                self.path['y'].insert(point[1])
            else:
                print("Point not inserted")
        else:
            self.path['x'].insert(point[0])
            self.path['y'].insert(point[1])

class watcher:
    def __init__(self, inBoundary = {'x': [], 'y': []}, outBoundary = {'x': [], 'y': []}):
        self.inBoundary = inBoundary
        self.outBoundary = outBoundary

    def insideBoundary(self, point, boundary):
        return (point[0] >= boundary['x'][0]) and (point[0] <= boundary['x'][1]) and (point[1] >= boundary['y'][0]) and (point[1] <= boundary['y'][1])

    def watch(self, path):
        vIn = self.gotIn(path)
        vOut = self.gotOut(path)
        if(vIn and vOut):
            return 'error'
        elif(vIn and not vOut):
            path.clean()
            return 'in'
        elif(not vIn and vOut):
            path.clean()
            return 'out'
        else:
            return 'none'


    def gotOut(self, path):
        if(path.isNotEmpty() and self.insideBoundary(path.lastPoint(), self.outBoundary)):
            outPoints = 0
            numberOfPoints = path.getLen()
            for i in range(1, numberOfPoints):
                if not(self.insideBoundary(path.indexedPoint(i), self.outBoundary)):
                    outPoints += 1
            return outPoints >= 1
        else:
            return False

    def gotIn(self, path):
        if(path.isNotEmpty() and self.insideBoundary(path.lastPoint(), self.inBoundary)):
            outPoints = 0
            numberOfPoints = path.getLen()
            for i in range(1, numberOfPoints):
                if not(self.insideBoundary(path.indexedPoint(i), self.inBoundary)):
                    outPoints += 1
            return outPoints >= 1
        else:
            return False