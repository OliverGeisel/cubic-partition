from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy, copy
from typing import List, Tuple

from conf.CubeGeneratorConf import CubeGeneratorConf
from conf.PointGeneratorConf import PointGeneratorConf
from conf.planeGeneratorConf import PlaneGeneratorConf
from core.point import random_Point, Point, GlobalPoint
from core.solution import ConcreteSolution, Partition


class Generator(ABC):

    def __init__(self, points=None, solution=None):
        self._created = False
        self.points_from_init = points if points is not None else list()
        self.solution_from_init = solution if points is not None else ConcreteSolution.empty_solution()
        if points is None:
            self.points = list()
        else:
            self.points = points
        if solution is None:
            self.correct = ConcreteSolution.empty_solution()
        else:
            self.correct = solution

    @abstractmethod
    def create_point_Instance(self, points: List[Point] = list(), correct: ConcreteSolution = None) -> Generator:
        pass

    def reset_generator(self):
        """
        WARNING if is used Global index is wrong
        :return:
        """
        self._created = False
        self.points = self.points_from_init
        self.correct = self.solution_from_init

    def plus(self, generator: Generator) -> Generator:
        """
               :param generator:
               :return: The other generator 
        """
        if not self._created:
            self.create_point_Instance()
        points, sol = self.get_instance_and_correct_solution()
        return generator.create_point_Instance(points, sol)

    def get_instance_and_correct_solution(self) -> Tuple[List[Point], ConcreteSolution]:
        return self.points, self.correct


class PointGenerator(Generator):

    def __init__(self, points, solution, conf: PointGeneratorConf = PointGeneratorConf.default_Conf()):
        super().__init__(points, solution)
        self.conf = conf

    def create_point_Instance(self, points: List[Point] = None,
                              correct: ConcreteSolution = None) -> Generator:
        # TODO fix partitions
        if self._created:
            return self
        if points is not None:
            self.points.extend(points)
        if correct is not None:
            self.correct = deepcopy(correct)
        all_points = [GlobalPoint(*random_Point().to_tuple()) for x in range(self.conf.amount)]
        self.points.extend(all_points)
        self.correct.change_instance(tuple(all_points))
        self._created = True
        return self


class CubeGenerator(Generator):

    def __init__(self, points=None, solution=None, conf: CubeGeneratorConf = CubeGeneratorConf.default_Conf()):
        super().__init__(points, solution)
        self.conf = conf

    def create_point_Instance(self, points: List[Point] = None, correct: ConcreteSolution = None) -> Generator:
        if self._created:
            return self
        if points is not None:
            self.points.extend(points)
        if correct is not None:
            self.correct = deepcopy(correct)
        all_points = list(self.correct.get_instance())
        for partition in range(self.conf.clusters):
            base_point = random_Point()
            new_partition = Partition(base_point)
            for run in range(self.conf.points_per_cluster):
                new_point = Point.spread_point(base_point, self.conf.max_spread)
                new_point = GlobalPoint(*new_point.to_tuple())
                new_partition.add(new_point)
                all_points.append(new_point)
                self.points.append(new_point)
            self.correct.partitions.append(new_partition)
        self.correct.change_instance(tuple(all_points))
        self.correct.size = len(all_points)
        self._created = True
        return self


# Todo add all following generaor the global points


class PlaneGenerator(Generator):

    def __init__(self, points=None, solution=None, conf: PlaneGeneratorConf = PlaneGeneratorConf.default_Conf()):
        super().__init__(points, solution)
        self.conf = conf

    @staticmethod
    def point_in_plane_by_normalvector(vector, origin: Point):
        pass

    @staticmethod
    def point_in_plane_by_vectors(origin: Point, vec1: Tuple[float, float, float], vec2: Tuple[float, float, float],
                                  scalefactor_one, scalefactor_two, normaliced=True):
        p1 = Point(*vec1)
        p2 = Point(*vec2)
        if normaliced:
            p1 = p1.get_normalized_point()
            p2 = p2.get_normalized_point()
        new_point = origin + p1 * random.uniform(*scalefactor_one) + p2 * random.uniform(*scalefactor_two)
        return new_point

    @staticmethod
    def point_in_plane_by_points(origin: Point, point1: Point, point2: Point, scalefactor_one: Tuple[float, float],
                                 scalefactor_two: Tuple[float, float], normaliced=True):
        vector1 = origin // point1
        vector2 = origin // point2
        # for easy use transfrom back to Point
        return PlaneGenerator.point_in_plane_by_vectors(origin, vector1, vector2, scalefactor_one, scalefactor_two,
                                                        normaliced)

    @staticmethod
    def create_normal_vector(origin, point1, point2) -> Tuple[float, float, float]:
        vector1 = origin // point1
        vector2 = origin // point2
        # back to point for operator use
        point_vector1 = Point(*vector1)
        point_vector2 = Point(*vector2)
        normal_vector = point_vector1.vector_product(point_vector2)
        return normal_vector

    def create_point_Instance(self, points: List[Point] = list(), correct: ConcreteSolution = None) -> Generator:
        if self._created:
            return self
        if points is not None:
            self.points.extend(points)
        if correct is not None:
            self.correct = deepcopy(correct)
        all_points = list(self.correct.get_instance())
        # create planes
        for index, partition in enumerate(range(self.conf.planes)):
            if len(self.conf.origins) <= index:
                index = len(self.conf.origins) - 1
            points_in_plane = self.conf.origins[index]
            origin = points_in_plane[0] if points_in_plane[0] is not None else random_Point()
            first_point = points_in_plane[1] if points_in_plane[1] is not None else random_Point()
            second_point = points_in_plane[2] if points_in_plane[2] is not None else random_Point()

            new_partition = Partition(origin)
            # run n times
            vector1 = Point(*(origin // first_point)).get_normalized_vector()
            vector2 = Point(*(origin // second_point)).get_normalized_vector()
            for run in range(self.conf.points_per_cluster):
                new_point = PlaneGenerator.point_in_plane_by_vectors(origin, vector1, vector2,
                                                                     self.conf.divergence, self.conf.divergence, False)
                if self.conf.divergence != 0.0:
                    new_point = Point.spread_point(new_point, self.conf.max_spread)
                    new_point = GlobalPoint(*new_point.to_tuple())
                new_partition.add(new_point)
                all_points.append(new_point)
                self.points.append(new_point)
            self.correct.partitions.append(new_partition)
        self.correct.change_instance(tuple(all_points))
        self.correct.size = len(all_points)
        self._created = True
        return self

    def add_plane(self, origin: Point, first_point: Point, second_point: Point):
        tmp = list(self.conf.origins)
        tmp.append((origin, first_point, second_point))
        self.conf.origins = tuple(tmp)
        self.reset_generator()


def AirCraftGenerator(Generator):
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
        return SphereGeneratorConf(1 << 7, 3, 1.0)


class SphereGenerator(Generator):

    def __init__(self, points=None, solution=None, conf=SphereGeneratorConf.default_Conf()):
        super().__init__(points, solution)
        self.conf = conf

    def create_point_Instance(self, points: List[Point] = None, correct: ConcreteSolution = None) -> Generator:
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
        all_points = list(self.correct.get_instance())
        for cluster in range(self.conf.clusters):
            base_point = random_Point()
            partition = Partition(base_point)
            for point in range(self.conf.points_per_cluster):
                cords = split_random_in_x_parts(self.conf.max_spread ** 2, 3)
                cords = [math.sqrt(x) * (-1) ** random.randint(0, 1) for x in cords]
                new_point = Point(*cords) + base_point
                new_point = GlobalPoint(*new_point.to_tuple())
                self.points.append(new_point)
                all_points.append(new_point)
                partition.add(new_point)
            self.correct.partitions.append(partition)
        self.correct.change_instance(tuple(all_points))
        self._created = True
        return self
