import math
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distance_between_two_points(self, pointB):
        return math.sqrt((self.x - pointB.getX())**2 + (self.y - pointB.getY())**2)
