from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np


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
        return math.sqrt(dist_x ** 2 + dist_y ** 2 + dist_z ** 2)

    def __mul__(self, other) -> Point:
        return Point(self.x * other, self.y * other, self.z * other)

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


class BidirectPoint(Point):
    def __init__(self, x=0.0, y=0.0, z=0.0, partition=None):
        super().__init__(x, y, z)
        self.__partition = partition

    def get_partition(self):
        return self.__partition


def random_Point(min_x=0, max_x=10, min_y=0, max_y=10, min_z=0, max_z=10):
    return Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y), random.uniform(min_z, max_z))
