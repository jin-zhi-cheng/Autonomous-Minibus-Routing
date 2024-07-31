import osmnx as ox
import csv
import numpy as np
import random
from shapely.geometry import Point, Polygon
import networkx as nx
import transbigdata as tbd
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import re

# POI list
def read_csv(file_name):
    poi_data = []
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            location = row[7].split(',')
            lon, lat = float(location[0]), float(location[1])
            poi_data.append({'value': float(row[5]), 'lon': lon, 'lat': lat})
    return poi_data

def calculate_weight(G, pois, radius=500):
    area1 = math.pi * radius ** 2
    para1 = 1 / area1
    area2 = math.sqrt(2 * math.pi)
    para2 = 1 / area2

    weights = {}
    for edge in G.edges(data=True):
        poi_count1 = 0
        poi_count2 = 0
        poi_count3 = 0
        node1, node2, data = edge
        node1_data = G.nodes[node1]
        node1_lat = node1_data['y']
        node1_lon = node1_data['x']
        node2_data = G.nodes[node1]
        node2_lat = node2_data['y']
        node2_lon = node2_data['x']

        for poi in pois:

            distance1 = ox.distance.great_circle_vec(node1_lat, node1_lon, poi['lat'], poi['lon'], earth_radius=6371009)
            distance2 = ox.distance.great_circle_vec(node2_lat, node2_lon, poi['lat'], poi['lon'], earth_radius=6371009)
            distance3 = ox.distance.great_circle_vec((node1_lat + node2_lat) / 2, (node1_lon + node2_lon) / 2,
                                                     poi['lat'], poi['lon'], earth_radius=6371009)

            if distance1 <= radius:
                poi_count1 += poi['value'] * para1 * para2 * math.exp((-1 * distance1 ** 2) / (2 * radius ** 2))
            else:
                poi_count1 += 0
            if distance2 <= radius:
                poi_count2 += poi['value'] * para1 * para2 * math.exp((-1 * distance2 ** 2) / (2 * radius ** 2))
            else:
                poi_count2 += 0
            if distance3 <= radius:
                poi_count3 += poi['value'] * para1 * para2 * math.exp((-1 * distance3 ** 2) / (2 * radius ** 2))
            else:
                poi_count3 += 0
        weights[(node1, node2)] = poi_count1 + poi_count2 + poi_count3
    return weights


def generate_random_coordinates(num_points):
    coordinates = [[min_lon, max_lat], [max_lon, max_lat], [max_lon, min_lat], [min_lon, min_lat]]
    random_coords = []
    polygon = Polygon(coordinates)

    while len(random_coords) < num_points:
        lng = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)
        point = Point(lng, lat)

        if polygon.contains(point):
            random_coords.append([lng, lat])

    return random_coords


def generate_risk(coords1, coords2, G, weights1):
    nearest_node1 = ox.distance.nearest_nodes(G, X=[coords1[0]], Y=[coords1[1]], return_dist=False)
    nearest_node2 = ox.distance.nearest_nodes(G, X=[coords2[0]], Y=[coords2[1]], return_dist=False)
    try:
        shortest_path = nx.shortest_path(G, nearest_node1[0], nearest_node2[0], weight='length')
        shortest_path_risk = 0
        for i in range(len(shortest_path) - 1):
            shortest_path_risk += weights1[(shortest_path[i], shortest_path[i + 1])]
    except nx.NetworkXNoPath:
        shortest_path_risk = 100

    return round(shortest_path_risk, 3)


def generate_risk_mat(coords, num, G, weights1):
    n = num + 1
    symmetric_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                symmetric_matrix[i, j] = 0
            else:
                symmetric_matrix[i, j] = generate_risk(coords[i], coords[j], G, weights1)
    for i in range(1, n):
        for j in range(0, i):
            symmetric_matrix[i, j] = symmetric_matrix[j, i]

    return symmetric_matrix

def coords_to_req_coords(coords, n):
    file_path = os.path.join('datatest', 'newmatrix%s_%s.txt' % (n, 120))
    data_path = os.path.abspath(file_path)
    with open(data_path) as f:
        lines = f.readlines()
        req = []
        req_coords = []
        labels = 1
        for i in range(labels, labels + n + 1):
            parts = re.split('[\t \n]', lines[i])
            req.append(int(parts[1]))
            req_coords.append(coords[req[i-1]])
    return req_coords