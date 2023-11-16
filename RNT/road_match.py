# -*- encoding=utf-8 -*-
#
# 最邻近匹配算法
from shapely.geometry import LineString, Point
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from geopy.distance import geodesic


def _query_node_related_roads(node_num, roads):
    node_related_roads = [list() for i in range(node_num)]
    for i in range(len(roads)):
        s, e = roads[i][0], roads[i][1]
        node_related_roads[s].append(i)
        node_related_roads[e].append(i)
    return node_related_roads


def _search_knn_nodes(nodes, xys):
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(nodes)
    distances, indices = nbrs.kneighbors(xys)
    return indices


def _search_geo_knn_nodes(nodes, xys):
    pass


def _roads_2_geometry(nodes, roads):
    roads_geometry = list()
    for road in roads:
        s, e = road[0], road[1]
        x1, y1 = nodes[s]
        x2, y2 = nodes[e]
        road_geometry = LineString([(x1, y1), (x2, y2)])
        roads_geometry.append(road_geometry)
    return roads_geometry


def _nodes_2_related_roads(nearest_nodes, node_related_roads):
    roads = list()
    for nodes in nearest_nodes:
        road = list()
        for node in nodes:
            road.extend(node_related_roads[node])
        roads.append(list(set(road)))
    return roads


def _spatial_dist(p1, p2):
    x = p1[0] - p2[0]
    y = p1[1] - p2[1]
    return (x * x + y * y) ** 0.5


def approximate_nearest_match(nodes, roads, xys):
    """
    该方法只针对每个road都是直线的情况下进行匹配，采用近似匹配的方法
    近似匹配：计算xy 最邻近的K个nodes,进而查找最邻近nodes相应roads计算距离，进而进行匹配
    :param nodes: 路网节点，列表，每一个元素为[x, y]
    :param roads: 路段，列表，每个元素为[start_node_id, end_node_id, length]
    :param xys: 坐标
    :return:
    """
    node_related_roads = _query_node_related_roads(len(nodes), roads)
    nearest_nodes = _search_knn_nodes(nodes, xys)
    nearest_roads = _nodes_2_related_roads(nearest_nodes, node_related_roads)
    roads_geometry = _roads_2_geometry(nodes, roads)
    match_points = list()
    for i in range(len(xys)):
        if i % 1000 == 0:
            print(i)
        p = Point(xys[i][0], xys[i][1])
        match_id = -1
        match_dist = 5000
        for road_id in nearest_roads[i]:
            dist = p.distance(roads_geometry[road_id])
            if dist < match_dist:
                match_id = road_id
                match_dist = dist
        assert match_id >= 0
        line = roads_geometry[match_id]
        np = line.interpolate(line.project(p))
        x, y = np.xy
        x = x[0]
        y = y[0]

        s, e = roads[match_id][0], roads[match_id][1]
        sd = _spatial_dist([x, y], nodes[s])
        ed = _spatial_dist([x, y], nodes[e])
        match_points.append([match_id, sd, ed, x, y])
    return match_points


# 针对大地坐标系进行的匹配
def approximate_geo_nearest_match(nodes, roads, xys):
    """
    该方法只针对每个road都是直线的情况下进行匹配，采用近似匹配的方法
    近似匹配：计算xy 最邻近的K个nodes,进而查找最邻近nodes相应roads计算距离，进而进行匹配
    :param nodes: 路网节点，列表，每一个元素为[x, y]
    :param roads: 路段，列表，每个元素为[start_node_id, end_node_id, length]
    :param xys: 坐标
    :return:
    """
    node_related_roads = _query_node_related_roads(len(nodes), roads)
    nearest_nodes = _search_knn_nodes(nodes, xys)
    nearest_roads = _nodes_2_related_roads(nearest_nodes, node_related_roads)
    roads_geometry = _roads_2_geometry(nodes, roads)
    match_points = list()
    for i in tqdm(range(len(xys)), desc='matching'):
        p = Point(xys[i][0], xys[i][1])
        match_id = -1
        match_dist = 5000
        for road_id in nearest_roads[i]:
            dist = p.distance(roads_geometry[road_id])
            if dist < match_dist:
                match_id = road_id
                match_dist = dist
        assert match_id >= 0
        line = roads_geometry[match_id]
        np = line.interpolate(line.project(p))
        x, y = np.xy
        x = x[0]
        y = y[0]

        s, e = roads[match_id][0], roads[match_id][1]
        sd = geodesic([y, x], nodes[s][::-1]).meters
        ed = geodesic([y, x], nodes[e][::-1]).meters
        sd = round(sd, 3)
        ed = round(ed, 3)
        match_points.append([match_id, sd, ed, x, y])
    return match_points
