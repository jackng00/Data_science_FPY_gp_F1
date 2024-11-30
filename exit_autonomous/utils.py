from os import chdir, system
from sklearn.cluster import KMeans
from numpy import array, asarray, max, vstack
from math import sqrt, tan, degrees
from open3d.cpu.pybind.io import read_point_cloud
from pandas import DataFrame
from pyntcloud import PyntCloud
import scipy.cluster.hierarchy as hcluster
import os
from orb_process import *
from time import sleep
from matplotlib import pyplot
from open3d.cpu.pybind.visualization import draw_geometries
from matplotlib.patches import Rectangle
from exit_process import *



# reads csv file and returns x, y, z arrays
def readCSV(fileName: str):
    file = open(fileName, 'r')
    x, y, z = [], [], []
    for line in file:
        line = line.strip('\n')
        arr = line.split(',')
        x.append(float(arr[0]))
        y.append(float(arr[1]))
        z.append(float(arr[2]))
    file.close()
    return x, y, z






# returns the x, y, z of all points the the point cloud
def pcdToArrays(pcd):
    pointsArray = list(asarray(pcd.points))
    x, y, z = [], [], []
    for point in pointsArray:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])
    return x, y, z


# make point cloud from xyz coordinates
def makeCloud(x, y, z):
    points = vstack((x, y, z)).transpose()
    cloud = PyntCloud(DataFrame(data=points, columns=["x", "y", "z"]))
    cloud.to_file("./PointData/output.ply")

    cloud = read_point_cloud("./PointData/output.ply")  # Read the point cloud
    return cloud


# returns coordinates of all the points outside the box
def pointsOutOfBox(x, y, box):
    bottomLeft = box[0]
    topRight = box[2]

    pointsX = []
    pointsY = []

    for i in range(len(x)):
        if bottomLeft[0] <= x[i] <= topRight[0] and bottomLeft[1] <= y[i] <= topRight[1]:
            continue
        pointsX.append(x[i])
        pointsY.append(y[i])

    return pointsX, pointsY


def hierarchicalClustering(x, y, thresh=1.5):
    points = []
    for i in range(len(x)):
        points.append([x[i], y[i]])

    data = array(points)

    # clustering
    clustersIndex = hcluster.fclusterdata(data, thresh, criterion="distance")

    clustersIndex = list(clustersIndex)

    numOfClusters = max(clustersIndex)

    clusters = [[] for _ in range(numOfClusters)]

    for i in range(len(points)):
        index = clustersIndex[i] - 1
        clusters[index].append(points[i])

    return clusters


# get the center of the clusters
def getClustersCenters(clusters):
    centerPoints = []
    for cluster in clusters:
        sumX, sumY = 0, 0
        for point in cluster:
            sumX += point[0]
            sumY += point[1]
        centerPoint = (float(sumX / len(cluster)), float(sumY / len(cluster)))
        centerPoints.append(centerPoint)
    return centerPoints


def getInlierOutlier(pcd, ind):
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    return inlier_cloud, outlier_cloud


def voxelDown(pcd, voxel_size=0.02):
    return pcd.voxel_down_sample(voxel_size=voxel_size)


# Every Kth points are selected
def selectEveryKPoints(pcd, K=5):
    return pcd.uniform_down_sample(every_k_points=K)


def removeStatisticalOutlier(pcd, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0):
    voxel_down_pcd = voxelDown(pcd, voxel_size)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return getInlierOutlier(voxel_down_pcd, ind)


def removeRadiusOutlier(pcd, voxel_size=0.02, nb_points=16, radius=0.05):
    voxel_down_pcd = voxelDown(pcd, voxel_size)
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return getInlierOutlier(voxel_down_pcd, ind)


# scatter 2d points with rectangle on the plot
def plot2DWithBox(x, y, box):
    bottomLeft = box[0]
    topRight = box[2]
    pyplot.scatter(x, y)
    width = topRight[0] - bottomLeft[0]
    height = topRight[1] - bottomLeft[1]
    rect = Rectangle((bottomLeft[0], bottomLeft[1]), width, height,
                     fill=False,
                     color="purple",
                     linewidth=2)
    pyplot.gca().add_patch(rect)
    pyplot.show()


# scatter all points in 2d and color the clusters centers in different color,
# also colors the furthest center in a third color
def plot2DWithClustersCenters(x, y, centers):
    pyplot.scatter(x, y)

    centersX = []
    centersY = []
    avgX = 0
    avgY = 0
    for coordinates in centers:
        avgX += coordinates[0]
        avgY += coordinates[1]
        centersX.append(coordinates[0])
        centersY.append(coordinates[1])
    pyplot.scatter(centersX, centersY)

    avgX = float(avgX / len(centers))
    avgY = float(avgY / len(centers))

    pyplot.scatter(avgX, avgY)
    print('Center of clusters centers: ', [avgX, avgY])

    center = avgX, avgY
    maxDistance = float('-inf')
    furthestCenter = None
    for point in centers:
        distance = distanceBetween2Points(center, point)
        if distance > maxDistance:
            maxDistance = distance
            furthestCenter = point

    print('furthestPoint: ', furthestCenter)
    pyplot.scatter(furthestCenter[0], furthestCenter[1])

    pyplot.show()


# gets x, y, z coordinates and shows an interactive points cloud
def showCloud(x, y, z):
    cloud = makeCloud(x, y, z)
    draw_geometries([cloud])  # Visualize the point cloud
