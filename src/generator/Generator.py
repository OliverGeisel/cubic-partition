from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy, copy
from typing import List, Tuple

import core
from conf.CubeGeneratorConf import CubeGeneratorConf
from conf.PointGeneratorConf import PointGeneratorConf
from core.Point import random_Point, Point
from core.Solution import Solution, Partition


class Generator(ABC):

    def __init__(self, points=None, solution=None):
        if points is None:
            self.points = list()
        else:
            self.points = points
        if solution is None:
            self.correct = Solution.empty_solution()
        else:
            self.correct = solution

    @abstractmethod
    def create_point_Instance(self, points: List[Point] = None, correct: Solution = None, **kwargs) -> Generator:
        pass

    @abstractmethod
    def plus(self, generator: Generator, **kwargs) -> Generator:
        """
               :param generator:
               :return: The other generator 
        """
        return generator.create_point_Instance(points=self.points, correct=self.correct, **kwargs)

    def get_instance_and_correct_solution(self) -> Tuple[List[Point], Solution]:
        return self.points, self.correct


class PointGenerator(Generator):

    def __init__(self, conf: PointGeneratorConf = PointGeneratorConf.default_Conf()):
        super().__init__()
        self.conf = conf

    def create_point_Instance(self, amount: int = 1 << 8, points: List[Point] = None,
                              correct: Solution = None) -> Generator:
        if points is not None:
            self.points.extend(copy(points))
        if correct is not None:
            self.correct = deepcopy(correct)

        self.points.extend([random_Point() for x in range(amount)])
        return self

    def plus(self, generator: Generator, **kwargs) -> Generator:
        return generator.create_point_Instance(points=self.points, correct=self.correct, **kwargs)


class CubeGenerator(Generator):

    def __init__(self, conf: PointGeneratorConf = CubeGeneratorConf.default_Conf()):
        super().__init__()
        self.conf = conf

    def create_point_Instance(self, points_per_cluster: int = 1 << 8, clusters: int = 5, max_spread: float = 0.5,
                              points: List[Point] = None, correct: Solution = None) -> Generator:
        if points is not None:
            self.points.extend(copy(points))
        if correct is not None:
            self.correct = deepcopy(correct)

        for partition in range(clusters):
            base_point = random_Point()
            new_partition = Partition(base_point)
            for run in range(points_per_cluster):
                new_partition.add(Point.spread_point(base_point, max_spread))
            self.correct.partitions.append(new_partition)
        return self

    def plus(self, generator: Generator, **kwargs) -> Generator:
        return generator.create_point_Instance(points=self.points, correct=self.correct, **kwargs)


class PlaneGenerator(Generator):
    # TODO     implement
    # Need  a Vector initializer
    #      a random variance for error while draw
    pass


def split_random_in_x_parts(val: float, x: int):
    old = val
    for y in range(x - 1):
        new = old * random.random()
        yield new
        old -= new
    yield old


class SphereGenerator(Generator):

    def create_point_Instance(self, points_per_cluster: int = 1 << 8, clusters: int = 5, max_spread: float = 1.0,
                              points: List[Point] = None, correct: Solution = None) -> Generator:
        """

        :param correct:
        :param points:
        :param points_per_cluster: NUmber of points in one Cluster
        :param clusters: number of Partitions, that are generated
        :param max_spread: the distance from the center-point of the partition
        :return: A list of points and a correct Solution for this list
        """
        if points is not None:
            self.points.extend(copy(points))
        if correct is not None:
            self.correct = deepcopy(correct)
        self.correct.size += clusters * points_per_cluster
        for cluster in range(clusters):
            base_point = random_Point()
            partition = Partition(base_point)
            for point in range(points_per_cluster):
                cords = split_random_in_x_parts(max_spread ** 2, 3)
                cords = [math.sqrt(x) * (-1) ** random.randint(0, 1) for x in cords]
                new_point = Point(*cords) + base_point
                self.points.append(new_point)
                partition.add(new_point)
            self.correct.partitions.append(partition)
        return self

    def plus(self, generator: Generator, **kwargs) -> Generator:
        return generator.create_point_Instance(points=self.points, correct=self.correct, **kwargs)
