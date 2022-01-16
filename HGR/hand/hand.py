import hand.point as pt

class Hand:
    def __init__(self, points):
        self.points = points
        if points is None:
            self.points = []


        length = len(self.points)
        for i in range(9-length):
            self.points.append(None)
