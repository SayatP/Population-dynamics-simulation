import numpy as np
import random

def two_int_to_hash(i1, i2):
    hashcode = 23
    hashcode = (hashcode * 37) + i1
    hashcode = (hashcode * 37) + i2
    return hashcode

def numpy_arr_to_str(arr):
    return '|'.join(''.join(str(i) for i in row) for row in arr)

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def closest_number_2(array, x1, y1):
    target_value = 2
    closest_points = []
    min_distance = float('inf')

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] == target_value:
                distance = manhattan_distance((i, j), (x1, y1))
                if distance < min_distance:
                    min_distance = distance
                    closest_points = [(i, j)]
                elif distance == min_distance:
                    closest_points.append((i,j))

    return closest_points


def get_single_value(matrix, agent_coords):
        distances = dict()
        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                if x != agent_coords[0] or y != agent_coords[1]:
                    dist = abs(agent_coords[0]-x) + abs(agent_coords[1]-y)

                    if dist not in distances:
                        distances[dist] = [(x, y, matrix[x,y].item())]   # x-coord, y-coord, value
                    else:
                        distances[dist].append((x, y, matrix[x,y].item()))


        terminal = 0

        for i, key in enumerate(sorted(distances.keys())):
            item = distances[key]

            if i == 0 and 1 in [triplet[2] for triplet in item]:
                return 1
            
            for triplet in item:
                if triplet[2] == 2:
                    terminal = 2

                if triplet[2] == 1:
                    if tuple(agent_coords) in closest_number_2(matrix, triplet[0], triplet[1]):
                        return 1

        return terminal


def approximate_5x5_to_3x3(matrix):
    new_t = np.zeros((3,3), dtype=np.int64)
    new_t[1,1] = 2

    for i in range(3):
        for j in range(3):
            if i != 1 or j != 1:
                new_t[i,j] = get_single_value(matrix[i:i+3,j:j+3], (2-i,2-j))

    return new_t

