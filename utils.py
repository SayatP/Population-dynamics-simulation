def two_int_to_hash(i1, i2):
    hashcode = 23
    hashcode = (hashcode * 37) + i1
    hashcode = (hashcode * 37) + i2
    return hashcode
