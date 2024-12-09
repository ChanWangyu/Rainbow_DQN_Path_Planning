
import math
import numpy as np
from envs.entities.state import OrderState, CarState
import logging


def compute_distance(obj1, obj2):
    if type(obj1) in [list, tuple]:
        assert len(obj1) == 3
        A, B = obj1, obj2
    else:
        assert type(obj1) in [OrderState, CarState]
        A = obj1.px, obj1.py, obj1.pz
        B = obj2.px, obj2.py, obj2.pz
    return math.sqrt(
        (A[0] - B[0]) ** 2 +
        (A[1] - B[1]) ** 2 +
        (A[2] - B[2]) ** 2
    )

def w2db(x):
    return 10 * math.log(x, 10)

def db2w(x):
    return math.pow(10, x/10)


if __name__ == '__main__':
    print(db2w(-9))

