from math import log2, floor
from utils import *


# return a rectangle that bounds the room and corresponds to the walls
def getAverageRectangle(x, y):
    # get AVG "center" point
    centerX = float(sum(x) / len(x))
    centerY = float(sum(y) / len(y))
    centerPoint = (centerX, centerY)

    # get all points left of center, right, up and down
    leftX = [(x[i], y[i]) for i in range(len(x)) if x[i] < centerX]
    rightX = [(x[i], y[i]) for i in range(len(x)) if x[i] > centerX]
    upY = [(x[i], y[i]) for i in range(len(y)) if y[i] > centerY]
    downY = [(x[i], y[i]) for i in range(len(y)) if y[i] < centerY]

    # calculate the distance between each point and the center
    leftDistances = [distanceBetween2Points(point, centerPoint) for point in leftX]
    rightDistances = [distanceBetween2Points(point, centerPoint) for point in rightX]
    upDistances = [distanceBetween2Points(point, centerPoint) for point in upY]
    downDistances = [distanceBetween2Points(point, centerPoint) for point in downY]

    # get rectangle coordinates
    xLeft = centerX - (2 * float(sum(leftDistances) / len(leftDistances)))
    xRight = centerX + (2 * float(sum(rightDistances) / len(rightDistances)))
    yUp = centerY + (2 * float(sum(upDistances) / len(upDistances)))
    yDown = centerY - (2 * float(sum(downDistances) / len(downDistances)))

    return (xLeft, yDown), (xRight, yDown), (xRight, yUp), (xLeft, yUp)


def boundingBox(points):
    """returns a list containing the al points of the bounding box,
        starting from bottom left and going anti-clockwise
        """
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    for x, _, y in points:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    return (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)


# gets a box and all points
# returns abs(points outside - points inside)
def getBoxFitness(box, points):
    bottomLeft = box[0]
    topRight = box[2]
    fitness = 0

    for point in points:

        # check if on the top side or bottom
        if bottomLeft[0] <= point[0] <= topRight[0] and ((point[1] == bottomLeft[1]) or (point[1] == topRight[1])):
            continue
        # check if on the left or right
        if bottomLeft[1] <= point[1] <= topRight[1] and ((point[0] == topRight[0]) or (point[0] == bottomLeft[0])):
            continue

        # check if out or inside the box
        if bottomLeft[0] < point[0] < topRight[0] and bottomLeft[1] < point[1] < topRight[1]:
            fitness = fitness + 1
        else:
            fitness = fitness - 1

    return abs(fitness)

# returns distance between 2 2D-points
def distanceBetween2Points(point1, point2):
    deltaX = point1[0] - point2[0]
    deltaY = point1[1] - point2[1]
    return sqrt((deltaX * deltaX) + (deltaY * deltaY))