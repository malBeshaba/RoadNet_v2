import numpy as np
import scipy.spatial
R = 6367
N = 10000


def changedata(data):
    phi = np.deg2rad(data[:, 1])  # LAT
    theta = np.deg2rad(data[:, 0])  # LON
    data = np.c_[data, R * np.cos(phi) * np.cos(theta), R * np.cos(phi) * np.sin(theta), R * np.sin(phi)]
    return data


def dist_to_arclength(chord_length):
    central_angle = 2 * np.arcsin(chord_length / (2.0 * R))
    arclength = R * central_angle
    return arclength


def using_kdtree(tree, que_Data):
    que_Data = changedata(que_Data)
    distance, index = tree.query(que_Data[:, 2:5])
    return dist_to_arclength(distance), index


def set_ref_tree(ref_points):
    ref_data = changedata(ref_points)
    return scipy.spatial.cKDTree(ref_data[:, 2:5])
