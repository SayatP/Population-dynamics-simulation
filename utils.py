import numpy as np
import random

def two_int_to_hash(i1, i2):
    hashcode = 23
    hashcode = (hashcode * 37) + i1
    hashcode = (hashcode * 37) + i2
    return hashcode

def numpy_arr_to_str(arr):
    return '|'.join(''.join(str(i) for i in row) for row in arr)

def get_single_value(t, agent_coords):
        distances = dict()
        for x in range(t.shape[0]):
            for y in range(t.shape[1]):
                if x != agent_coords[0] or y != agent_coords[1]:
                    dist = abs(agent_coords[0]-x) + abs(agent_coords[1]-y)

                    if dist not in distances:
                        distances[dist] = [t[x,y].item()]
                    else:
                        distances[dist].append(t[x,y].item())

        for i, key in enumerate(sorted(distances.keys())):
            item = distances[key]

            if i == 0 and 1 in item:
                return 1
            
            if 1 in item or 2 in item:
                if item.count(2) > item.count(1):
                    return 2
                return 1

        return 0

def approximate_5x5_to_3x3(t):
    new_t = np.zeros((3,3), dtype=np.int64)
    new_t[1,1] = 2

    for i in range(3):
        for j in range(3):
            if i != 1 or j != 1:
                new_t[i,j] = get_single_value(t[i:i+3,j:j+3], (2-i,2-j))

    return new_t
