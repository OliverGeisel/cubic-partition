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
        self._created = False
        if points is None:
            self.points = list()
        else:
            self.points = points
        if solution is None:
            self.correct = Solution.empty_solution()
        else:
            self.correct = solution

    @abstractmethod
    def create_point_Instance(self, points: List[Point] = list(), correct: Solution = None) -> Generator:
        pass

    def reset_generator(self):
        self._created = False

    def plus(self, generator: Generator) -> Generator:
        """
               :param generator:
               :return: The other generator 
        """
        if not self._created:
            self.create_point_Instance()
        points, sol = self.get_instance_and_correct_solution()
        return generator.create_point_Instance(points, sol)

    def get_instance_and_correct_solution(self) -> Tuple[List[Point], Solution]:
        return self.points, self.correct


class PointGenerator(Generator):

    def __init__(self, points, solution, conf: PointGeneratorConf = PointGeneratorConf.default_Conf()):
        super().__init__(points, solution)
        self.conf = conf

    def create_point_Instance(self, points: List[Point] = None,
                              correct: Solution = None) -> Generator:
        # TODO fix partitions
        if self._created:
            return self
        if points is not None:
            self.points.extend(points)
        if correct is not None:
            self.correct = deepcopy(correct)
        all_points = [random_Point() for x in range(self.conf.amount)]
        self.points.extend(all_points)
        self.correct.complete_graph = tuple(all_points)
        self._created = True
        return self


class CubeGenerator(Generator):

    def __init__(self, points=None, solution=None, conf: CubeGeneratorConf = CubeGeneratorConf.default_Conf()):
        super().__init__(points, solution)
        self.conf = conf

    def create_point_Instance(self, points: List[Point] = None, correct: Solution = None) -> Generator:
        if self._created:
            return self
        if points is not None:
            self.points.extend(points)
        if correct is not None:
            self.correct = deepcopy(correct)
        all_points = list(self.correct.complete_graph)
        for partition in range(self.conf.clusters):
            base_point = random_Point()
            new_partition = Partition(base_point)
            for run in range(self.conf.points_per_cluster):
                new_point = Point.spread_point(base_point, self.conf.max_spread)
                new_partition.add(new_point)
                all_points.append(new_point)
                self.points.append(new_point)
            self.correct.partitions.append(new_partition)
        self.correct.complete_graph = tuple(all_points)
        self.correct.size = len(all_points)
        self._created = True
        return self


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


class SphereGeneratorConf(object):

    def __init__(self, points_per_cluster: int, clusters: int, max_spread: float):
        self.points_per_cluster = points_per_cluster
        self.clusters = clusters
        self.max_spread = max_spread

    @staticmethod
    def default_Conf():
        return SphereGeneratorConf(1 << 8, 5, 1.0, )

    pass


class SphereGenerator(Generator):

    def __init__(self, points=None, solution=None, conf=SphereGeneratorConf.default_Conf()):
        super().__init__(points, solution)
        self.conf = conf

    def create_point_Instance(self, points: List[Point] = None, correct: Solution = None) -> Generator:
        """

        :param correct:
        :param points:
        :param points_per_cluster: NUmber of points in one Cluster
        :param clusters: number of Partitions, that are generated
        :param max_spread: the distance from the center-point of the partition
        :return: A list of points and a correct Solution for this list
        """
        if self._created:
            return self
        if points is not None:
            self.points.extend(points)
        if correct is not None:
            self.correct = deepcopy(correct)
        self.correct.size += self.conf.clusters * self.conf.points_per_cluster
        all_points = list(self.correct.complete_graph)
        for cluster in range(self.conf.clusters):
            base_point = random_Point()
            partition = Partition(base_point)
            for point in range(self.conf.points_per_cluster):
                cords = split_random_in_x_parts(self.conf.max_spread ** 2, 3)
                cords = [math.sqrt(x) * (-1) ** random.randint(0, 1) for x in cords]
                new_point = Point(*cords) + base_point
                self.points.append(new_point)
                all_points.append(new_point)
                partition.add(new_point)
            self.correct.partitions.append(partition)
        self.correct.complete_graph = tuple(all_points)
        self._created = True
        return self
