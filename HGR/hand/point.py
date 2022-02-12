class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


    def slope(self, pt2):
        return (pt2.y-self.y)/(pt2.x-self.x)
