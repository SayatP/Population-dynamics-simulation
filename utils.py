import random

def two_int_to_hash(i1, i2):
    hashcode = 23
    hashcode = (hashcode * 37) + i1
    hashcode = (hashcode * 37) + i2
    return hashcode

def numpy_arr_to_str(arr):
    return '|'.join(''.join(str(i) for i in row) for row in arr)
