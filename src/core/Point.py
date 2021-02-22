from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np


class Point_np:

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.values = np.array((x, y, z), dtype=np.float32)

    def __abs__(self):
        return np.sqrt(np.sum(np.square(self.values)))

    def __sub__(self, other):
        dist = self.values - other.values
        return np.sqrt((dist * dist).sum())


class Bipoint_np(Point_np):
    def __init__(self, x, y, z, part):
        super().__init__(x, y, z)
        self._partition = part


class Point:
    """
        A class that represents the Point in 3D.
        Warning! Some operators are overwritten an have special behavior
    """

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def spread_point(from_point: Point, spread: float):
        x = np.random.uniform(from_point.x - spread, from_point.x + spread)
        y = np.random.uniform(from_point.y - spread, from_point.y + spread)
        z = np.random.uniform(from_point.z - spread, from_point.z + spread)
        return Point(x, y, z)

    def __abs__(self) -> float:
        """
        Distance to origin
        :return: distance as float value
        """
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __add__(self, other) -> Point:
        """
        Add to the three the values that are given by the other object
        :param other: a object with exact three components
        :return: a new Point with add values
        """
        if isinstance(other, Point):
            x = self.x + other.x
            y = self.y + other.y
            z = self.z + other.z
            return Point(x, y, z)
        elif len(other) == 3:
            x = self.x + other[0]
            y = self.y + other[1]
            z = self.z + other[2]
            return Point(x, y, z)
        else:
            raise Exception("The param 'other' (right of +) is not applicable on an a Point")

    def __sub__(self, other) -> float:
        """

        :param other: Second point to calc distance
        :return: (euclidean) distance between two points
        """
        dist_x = self.x - other.x
        dist_y = self.y - other.y
        dist_z = self.z - other.z
        return math.sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z)

    def __mul__(self, other) -> Point:
        return Point(self.x * other, self.y * other, self.z * other)

    def __matmul__(self, other) -> float:
        return self.x * other.x + self.y * other.y * self.z * other.z

    def __floordiv__(self, other) -> Tuple[float, float, float]:
        """
        Get the direction-Vector of the two points. Note left Point is source and right is target
        :param other:
        :return:
        """
        x = other.x - self.x
        y = other.y - self.y
        z = other.z - self.z
        return x, y, z

    def __truediv__(self, other) -> Point:
        """
           creates the middle Point of two Points
        :param other: second Point that is needed for middle point
        :return: middle Point
        """
        dist_x = (self.x + other.x) / 2
        dist_y = (self.y + other.y) / 2
        dist_z = (self.z + other.z) / 2
        return Point(dist_x, dist_y, dist_z)

    def __str__(self):
        return f"Point: x= {self.x} y= {self.y} z= {self.z}"

    def to_tuple(self):
        return self.x, self.y, self.z

    def vector_product(self, other: Point) -> Tuple[float, float, float]:
        x1 = self.y * other.z - self.z * other.y
        x2 = self.z * other.x - self.x * other.z
        x3 = self.x * other.y - self.y * other.x
        return x1, x2, x3

    def get_normalized_vector(self) -> Tuple[float, float, float]:
        absolute = abs(self)
        x = self.x / absolute
        y = self.y / absolute
        z = self.z / absolute
        return x, y, z

    def get_normalized_point(self):
        return Point(*self.get_normalized_vector())


class GlobalPoint(Point):
    __index_count = 0

    @staticmethod
    def __get_next_index():
        tmp = GlobalPoint.__index_count
        GlobalPoint.__index_count += 1
        return tmp

    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.index = GlobalPoint.__get_next_index()


class BidirectPoint(Point):
    def __init__(self, x=0.0, y=0.0, z=0.0, partition=None):
        super().__init__(x, y, z)
        self._partition = partition

    def get_partition(self):
        return self._partition

    def __str__(self):
        return super(BidirectPoint, self).__str__() + f"{self._partition}"


class BidiectPointEncode(Point):
    def __init__(self, x=0.0, y=0.0, z=0.0, partition_number=-1):
        super().__init__(x, y, z)
        self._partition = partition_number

    def get_partition(self):
        return self._partition

    def __str__(self):
        return super(Point, self).__str__() + f"{self._partition}"

    def to_tuple(self):
        return self.x, self.y, self.z, self._partition


def random_Point(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10):
    return Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y), random.uniform(min_z, max_z))
