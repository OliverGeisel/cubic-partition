from __future__ import annotations

import random
from copy import copy, deepcopy
from typing import List, Tuple

import numpy as np

from core.Point import Point


def default_init(solution):
    for point in solution.complete_graph:
        solution.partitions[random.randint(0, len(solution.partitions) - 1)].add(point)


class Solution:

    @staticmethod
    def empty_solution() -> Solution:
        return Solution(list(), 0)

    @staticmethod
    def solution_from_numpy_array(array: np.ndarray) -> Solution:
        shape = array.shape
        solution = Solution(list(), shape[0])
        partitions = list()
        for partition in array:
            new_partition = Partition()
            for point in partition:
                new_point = Point(*point)
                new_partition.add(new_point)
                solution.complete_graph.append(new_point)
            partitions.append(new_partition)
        solution.partitions = partitions
        # solution.complete_graph = [Point(*point) for pat in array for point in pat]
        return solution

    def __init__(self, instance: List[Point], partitions: int = 3, init_func=default_init, old_solution=None):
        self.size = len(instance)
        self.partitions = [Partition() for x in range(partitions)]
        self.complete_graph = instance
        self.__old_solution = old_solution
        init_func(self)
        for part in self.partitions:
            part.update_center()

    def __len__(self):
        return self.size

    def update_centers(self):
        for part in self.partitions:
            part.update_center()

    def get_centers(self) -> List[Point]:
        return [part.get_center() for part in self.partitions]

    def get_value(self) -> float:
        back = 0
        num_partitions = len(self.partitions)
        num_points = self.size

        for partition in self.partitions:
            cardi_partition = len(partition)
            sum_all_points = 0
            for point in partition.points_in_partition:
                sum_point = 0
                for other_point in partition.points_in_partition:
                    sum_point += point - other_point
                sum_all_points += sum_point
            back += (1 / num_partitions) * (cardi_partition / num_points) * sum_all_points
        return back

    def is_changed(self):
        for x in self.partitions:
            if x.changed:
                return True
        return False

    def to_numpy_array(self):
        """
            crate a numpy array of current solution
            the array has 3 dims.
            the shape is (clusters, num_in_culsters, cords )
            For example a Solution with 5 Clusters where each cluster have 100 Points look like (5,100,3)
        :return: numpy array of this solution
        """
        complete = [part.to_numpy_array() for part in self.partitions]
        return np.array(complete)

    def clone(self) -> Solution:
        clone = copy(self)
        clone.partitions = deepcopy(self.partitions)
        return clone




#### Partition


class Partition:

    # TODO reduce cals when updated/ not updated
    # distance onlyx update if changed

    def __init__(self, center: Point = Point()):
        self._center = center
        self.points_in_partition = list()
        self.changed = False

    def __abs__(self):
        back = 0
        for point in self.points_in_partition:
            back += point - self._center
        return back / len(self.points_in_partition)

    def __iter__(self):
        return self.points_in_partition.__iter__()

    def __len__(self):
        return len(self.points_in_partition)

    def __deepcopy__(self, memodict):
        return copy(self)

    def add(self, point: Point):
        self.points_in_partition.append(point)
        self.changed = True

    def remove(self, point: Point):
        self.points_in_partition.remove(point)
        self.changed = True

    def update_center(self):
        if not self.is_changed():
            return
        x_val = 0
        y_val = 0
        z_val = 0

        for point in self.points_in_partition:
            x_val += point.x
            y_val += point.y
            z_val += point.z
        num_points = len(self.points_in_partition)
        if num_points == 0:
            return
        self._center = Point(x_val / num_points, y_val / num_points, z_val / num_points)
        self.changed = False

    def get_variance(self) -> float:
        """
        is semantic equivalent to __abs__
        :return:
        """
        variance_sum = 0
        for p in self.points_in_partition:
            variance_sum += p - self._center
        return variance_sum / len(self.points_in_partition)

    def get_max_distance_point(self) -> Point:
        max_point = None
        max_dist = 0
        for p in self.points_in_partition:
            dist = p - self._center
            if dist > max_dist:
                max_dist = p - self._center
                max_point = p
        return max_point

    def get_as_three_lists(self):
        x = list()
        y = list()
        z = list()
        for point in self.points_in_partition:
            x.append(point.x)
            y.append(point.y)
            z.append(point.z)
        return x, y, z

    def split_to_two(self) -> Tuple[Partition, Partition]:
        max_dist = self.get_max_distance_point()
        # Todo factor for new center depending on variance to max point
        new_center1 = max_dist / self._center
        vector = new_center1 // self._center
        new_center2 = self._center + vector

        new_part1 = Partition(new_center1)
        new_part2 = Partition(new_center2)
        for p in self.points_in_partition:
            if p - new_center1 < p - new_center2:
                new_part1.add(p)
            else:
                new_part2.add(p)
        new_part2.update_center()
        new_part1.update_center()
        return new_part1, new_part2

    def to_numpy_array(self):
        complete = [point.to_tuple() for point in self.points_in_partition]
        return np.array(complete)

    def get_center(self) -> Point:
        return self._center

    def is_changed(self) -> bool:
        return self.changed
