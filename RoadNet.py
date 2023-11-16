import shapefile
from copy import copy
import networkx as nx
from shapely.geometry import Point, Polygon, LineString
from enum import Enum
from tqdm import tqdm
from RNT.netShape import transfer
import numpy as np
from geopy.distance import geodesic
from functools import cached_property
from RNT.match_kdtree import set_ref_tree, using_kdtree
import pandas as pd
from RNT.road_match import approximate_geo_nearest_match


class Relation(Enum):
    contain = 1
    all_contain = 2
    not_contain = 0


class Topo:
    def __init__(self, node, roadnet):
        try:
            self.lines = (node, roadnet.get_road(node), list(roadnet.get_record(node))[5])
        except IndexError:
            print(node)
            print(roadnet.get_record(node))
        self.head = self.set_struct(0, roadnet)
        self.tail = self.set_struct(len(self.lines[1]) - 1, roadnet)
        self.edges = [self.set_struct(index, roadnet) for index in range(1, len(self.lines[1]) - 1)]

    def set_struct(self, line_node, roadnet):
        """
        创建路网拓扑结构
        :param line_node: 路段结点index
        :param roadnet: 路网结构
        :return: 路网拓扑结构
        """
        road_node = roadnet.get_node_dict(self.lines[1][0 + line_node])
        # print(roadnet.get_node_list(road_node))
        edges = roadnet.get_edge_dict(tuple(roadnet.get_node_list(road_node)))
        # print(edges)
        node = roadnet.get_node_list(road_node)
        if len(edges) == 1:
            return []
        result = []
        for edge in edges:
            road = roadnet.get_road(edge)
            num = -1
            for index in range(len(road)):
                if node == road[index]:
                    num = index
                    if index == len(road) - 1:
                        num = -1
            # 连接的路id 连接的结点 连接的结点在连接的路上的位置
            result.append((edge, road_node, num))
        return result


class RoadNet:
    def __init__(self, shf, roads, records, node_list, node_dict, edge_dict, G, filename, road_dist, road_dist_D):
        self.__roads_distances = None
        self.topoList = None
        self.__shf = shf
        self.__roads = roads
        self.__records = records
        self.__node_list, self.__node_dict, self.__edge_dict, self.__G = node_list, node_dict, edge_dict, G
        self.__lines, self.__line_buffers = None, None
        self.__filename = filename
        self.__road_dist = road_dist
        self.__road_dist_D = road_dist_D

    @classmethod
    def load_by_cache(cls, road_path):
        """
        从缓存中加载路网
        :param road_path: 路网路径
        :return: RoadNet对象
        """
        shf = shapefile.Reader(road_path)
        roads = [shape.points for shape in shf.shapes()]
        records = [record for record in shf.records()]
        name = road_path.split('\\')[-1].replace('.shp', '')
        cache_dir = r'../TAZ/cache'
        node_dict = np.load(cache_dir + '/' + name + 'node_dict.npy', allow_pickle=True).item()
        node_list = np.load(cache_dir + '/' + name + 'node_list.npy').tolist()
        edge_dict = np.load(cache_dir + '/' + name + 'edge_dict.npy', allow_pickle=True).item()
        G = nx.Graph()
        for road in tqdm(roads, desc='Initializing'):
            for index in range(len(road) - 1):
                G.add_edge(node_dict[road[index]], node_dict[road[index + 1]])
        return cls(shf, roads, records, node_list, node_dict, edge_dict, G, road_path)

    @classmethod
    def load_by_shapefile(cls, road_path):
        """
        从shapefile中加载路网
        :param road_path: 路网路径
        :return: RoadNet对象
        """
        shf = shapefile.Reader(road_path, encoding='gb18030')
        roads = [shape.points for shape in shf.shapes()]
        records = [record for record in shf.records()]

        def Initialize(roads):
            node_list, node_dict, edge_dict = [], {}, {}
            G = nx.Graph()

            def add_node(n):
                try:
                    node_dict[n]
                except KeyError:
                    node_dict[n] = len(node_list)
                    node_list.append(n)
                # if n not in node_list:
                #     node_dict[n] = len(node_list)
                #     node_list.append(n)
                return node_dict[n]

            def add_edge(n, num):
                try:
                    edge_dict[n]
                except KeyError:
                    edge_dict[n] = [num]
                edge_dict[n].append(num)
                # if n not in edge_dict.keys():
                #     edge_dict[n] = [num]
                # else:
                #     edge_dict[n].append(num)

            road_num = 0
            # sw = shapefile.Writer(r'D:\data\路网\osmBA\circle_0.shp', shapeType=shapefile.POINT)
            # sw.field('ID', 'C')
            segments_distanse = []
            road_dist_D = {}
            for road in tqdm(roads, desc='Initializing'):
                # 判断成环路
                # if len(road) > 1 and road[0] == road[-1]:
                #     x, y = road[0]
                #     sw.point(x, y)
                #     sw.record(str(road_num))
                #     road_num += 1
                #     print('Circle Error: ', road[0], road[-1], 'in road', road_num, road)
                #     continue
                for index in range(len(road) - 1):
                    # 初始化时增加验证，防止出现重复的点
                    # if road[index] == road[index + 1]:
                        # print('Repeat Error: ', road[index], road[index + 1], 'in road', road_num,)
                        # continue
                    G.add_edge(add_node(road[index]), add_node(road[index + 1]))
                    road_dist_D[len(segments_distanse)] = road_num
                    segments_distanse.append([add_node(road[index]), add_node(road[index + 1]), geodesic(road[index][::-1], road[index + 1][::-1]).meters])
                    add_edge(road[index], road_num)
                add_edge(road[-1], road_num)
                road_num += 1
            # sw.close()
            return node_list, node_dict, edge_dict, G, segments_distanse, road_dist_D

        node_list, node_dict, edge_dict, G, road_dist, road_dist_D = Initialize(roads)
        return cls(shf, roads, records, node_list, node_dict, edge_dict, G, road_path, road_dist, road_dist_D)

    def with_distances(self):
        """
        计算路网中每条路的长度
        :return: self
        """
        distance = self.__distances
        L, i = [], 0
        for road in self.__roads:
            first_index = self.__node_dict[road[0]]
            last_index = self.__node_dict[road[-1]]
            L.append([first_index, last_index, distance[i]])
            i += 1
        self.__roads_distances = L
        return self

    @cached_property
    def __distances(self):
        """
        Calculate the length of each road
        :return: [Road distance]
        """
        L = []
        for road in tqdm(self.__roads, desc='Calculating distances'):
            meter = 0.0
            for index in range(len(road) - 1):
                meter += geodesic(road[index][::-1], road[index + 1][::-1]).meters
            L.append(meter)
        return L

    def get_roads(self):
        """
        获取路网中的路段距离
        :return: 路段距离列表，头节点、尾结点、距离
        """
        return self.__road_dist

    def get_fields(self):
        """
        获取路网中的字段
        :return:  字段列表
        """
        return self.__shf.fields

    def get_shapes(self):
        """
        获取路网中的形状
        :return:  形状列表
        """
        self.__roads = [road for road in self.__roads if road]
        self.__records = [record for record in self.__records if record]
        return [(self.__roads[index], self.__records[index]) for index in range(len(self.__roads))]

    def get_road_distances(self):
        return self.__roads_distances

    def get_road(self, num):
        return self.__roads[num]

    def get_record(self, num):
        return self.__records[num]

    def get_node_list(self, num):
        return self.__node_list[num]

    def get_nodes(self):
        return self.__node_list

    def get_node_dict(self, key):
        return self.__node_dict[key]

    def get_edge_dict(self, key):
        return self.__edge_dict[key]

    def prun_loop(self):
        net = copy(self)
        index = 1

        # 存在一定问题，存在少数线头删除不成功的情况
        def prun():
            remove_nodes = []
            for node in net.__G.nodes():
                if net.__G.degree(node) == 1:
                    remove_nodes.append(node)
            return remove_nodes

        sw = shapefile.Writer(r'D:\data\路网\osmBA\point_one_edge4.shp', shapeType=shapefile.POINT)
        sw.field('ID', 'C')

        while True:
            rns = prun()

            if len(rns) == 0:
                break
            r = 0
            title = 'prun ' + str(index)
            for node in tqdm(rns, desc=title):
                x, y = net.__node_list[node]
                sw.point(x, y)
                sw.record(str(node))
                edge = net.__edge_dict[tuple(net.__node_list[node])]
                # if len(self.__records[edge[0]]) == 0:
                #     r += 1
                #     continue
                L = []
                for point in net.__roads[edge[0]]:
                    if point != tuple(net.__node_list[node]):
                        L.append(point)
                net.__roads[edge[0]] = L
                # net.__records[edge[0]] = []
                net.__edge_dict[tuple(net.__node_list[node])] = []
                net.__G.remove_node(node)
            index += 1
            # if r == len(rns):
            #     for node in rns:
            #         net.__G.remove_node(node)
        sw.close()
        return net

    def topology(self):
        tp_list = []
        for index in tqdm(range(len(self.__roads))):
            tp_list.append(Topo(index, self))
        self.topoList = tp_list

    def set_buffer(self):
        lines = []
        line_buffers = []
        for road in self.__roads:
            lines.append(LineString(road))
            line_buffers.append(LineString(road).buffer(0.0005))
        return lines, line_buffers

    def get_line_relation(self, line1, line2):
        num = 0
        if self.__line_buffers[line1].contains(self.__lines[line2]):
            num += 1
        if self.__line_buffers[line2].contains(self.__lines[line1]):
            num += 1
        return Relation(num)

    def toShapefile(self, save_url):
        transfer(self, save_url)

    def save_cache(self):
        name = self.__filename.split('\\')[-1].replace('.shp', '')
        cache_dir = r'../RNT/cache'
        np.save(cache_dir + '/' + name + 'roads.npy', np.array(self.__roads))
        np.save(cache_dir + '/' + name + 'node_list.npy', np.array(self.__node_list))
        np.save(cache_dir + '/' + name + 'node_dict.npy', self.__node_dict, allow_pickle=True)
        np.save(cache_dir + '/' + name + 'edge_dict.npy', self.__edge_dict, allow_pickle=True)
        nx.write_gexf(self.__G, cache_dir + '/' + name + 'group.gexf')

    def delete_roads_by_type(self, type_id):
        net = copy(self)

        def prun(road_id):
            road = net.get_road(road_id)
            num = 0
            for index in range(len(road) - 1):
                net.__G.remove_edge(self.__node_dict[road[index]], self.__node_dict[road[index + 1]])
                num += 1
            return num

        def is_closed_loop():
            solo_nodes = []
            for node in net.__G.nodes():
                if net.__G.degree(node) == 1:
                    solo_nodes.append(node)
            return solo_nodes

        def find_topo(node):
            roads = self.__edge_dict[tuple(self.__node_list[node])]
            road = roads[0]
            for r in roads:
                if self.__records[r][5] != type_id:
                    road = r
                    break
            topo = net.topoList[road]
            if tuple(self.__node_list[node]) == self.get_road(road)[0]:
                if len(topo.head) == 0:
                    return -1, -1, -1
                line = net.topoList[topo.head[0][0]]
                isHead = True
            else:
                line = net.topoList[topo.tail[0][0]]
                isHead = False
            while net.topoList[line.tail[0][0]].lines[2] != type_id:
                line = net.topoList[topo.tail[0][0]]
            return road, isHead, line.tail[0]

        num = 0
        for index in range(len(self.__records)):
            if self.__records[index][5] == type_id:
                num += prun(index)
        print('删除类型' + str(type_id) + '路段共' + str(num) + '条')
        solo_nodes = is_closed_loop()
        for node in solo_nodes:
            road, isHead, topo_l = find_topo(node)
            if road == -1:
                continue
            if isHead:
                self.__roads[road].insert(0, self.__roads[topo_l[0]][topo_l[2]])
            else:
                self.__roads[road].append(self.__roads[topo_l[0]][topo_l[2]])

    def point_on_line(self, dist):
        """
        Get points on an equidistant tangent line
        :param dist: int (At what distance)
        :return: RoadNet()
        """

        def existPoint(m):
            return True if m > dist else False

        for roadIndex in tqdm(range(len(self.__roads)), desc='pointing'):
            if existPoint(self.__distances[roadIndex]):
                nodes = self.__roads[roadIndex]
                for n in range(len(nodes) - 1):
                    meter = geodesic(nodes[n][::-1], nodes[n + 1][::-1]).meters
                    if existPoint(meter):
                        for i in range(int(dist / meter)):
                            span_x = nodes[n][0] - nodes[n][0]
                            span_y = nodes[n + 1][1] - nodes[n + 1][1]
                            norm_value = (span_x * span_x + span_y * span_y) ** 0.5
                            newNodeLon = nodes[n][0] + span_x * dist / norm_value
                            newNodeLat = nodes[n][1] + span_y * dist / norm_value
                            self.__node_dict[(newNodeLon, newNodeLat)] = [len(self.__node_list)]
                            self.__edge_dict[len(self.__node_list)] = roadIndex
                            self.__node_list.append([newNodeLon, newNodeLat])
                            # self.__roads[roadIndex].insert(n+i, (newNodeLon, newNodeLat))
        return self

    # def merge_mutiline(self, type_id):

    def match_on_road_ckdtree(self, points, save_path=''):
        """
        Match points on road
        :param save_path:
        :param points: list of points
        :return: list of road index
        """
        tree = set_ref_tree(np.array(self.__node_list))
        dist_ckdn, indexes_ckdn = using_kdtree(tree, np.array(points))
        print(dist_ckdn, indexes_ckdn)
        m_points = [self.__node_list[i] for i in indexes_ckdn]
        m_roads = [self.__edge_dict[i][0] for i in m_points]
        print(m_points, m_roads)
        start_road_point = [self.__node_list[self.__roads_distances[m_roads[i]][0]] for i in range(len(m_roads))]
        end_road_point = [self.__node_list[self.__roads_distances[m_roads[i]][1]] for i in range(len(m_roads))]
        start_distance = [geodesic(m_points[i][::-1], start_road_point[i][::-1]).meters for i in range(len(m_points))]
        end_distance = [geodesic(m_points[i][::-1], end_road_point[i][::-1]).meters for i in range(len(m_points))]
        match_roads = [[m_roads[i], start_distance[i], end_distance[i], m_points[i][0], m_points[i][1]] for i in
                       range(len(m_points))]
        if save_path != '':
            pd.DataFrame(match_roads, columns=['rid', 'sdist', 'edist', 'lng', 'lat']).to_csv(save_path, index=False, header=False)
            # np.save(save_path, np.array(match_roads))
        return match_roads

    def match_on_roads_nearest(self, points, save_path=''):
        """
        Match points on road
        :param save_path:
        :param points: list of points
        :return: list of road index
        """
        match_roads = approximate_geo_nearest_match(self.__node_list, self.__road_dist, points)
        m_roads = [self.__road_dist_D[mr[0]] for mr in match_roads]
        start_road_point = [self.__roads[i][0] for i in m_roads]
        end_road_point = [self.__roads[i][-1] for i in m_roads]
        m_points = [(mr[3], mr[4]) for mr in match_roads]
        start_distance = [geodesic(m_points[i][::-1], start_road_point[i][::-1]).meters for i in range(len(m_points))]
        end_distance = [geodesic(m_points[i][::-1], end_road_point[i][::-1]).meters for i in range(len(m_points))]
        match_roads = [[m_roads[i], start_distance[i], end_distance[i], m_points[i][0], m_points[i][1]] for i in range(len(m_points))]
        if save_path != '':
            pd.DataFrame(match_roads, columns=['rid', 'sdist', 'edist', 'lng', 'lat']).to_csv(save_path, index=False, header=False)
        return match_roads
