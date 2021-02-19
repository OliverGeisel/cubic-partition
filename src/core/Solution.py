from __future__ import annotations

import random
from copy import copy, deepcopy
from typing import List, Tuple, Dict

import numpy as np

from core.Point import Point, BidirectPoint


def default_init(solution):
    for point in solution.complete_graph:
        index = random.randint(0, len(solution.partitions) - 1)
        solution.partitions[index].add(point)


class Solution:

    @staticmethod
    def empty_solution() -> Solution:
        """
        Creates an empty Solution wirh no Point and 0 Partitions
        :return: an new empty Solution
        """
        return Solution(tuple(), 0)

    @staticmethod
    def solution_from_numpy_array(array: np.ndarray) -> Solution:
        """
        Parse a Solution from a numpy-array, if the shape is matching
        :param array: the array with
        :return: the Solution that contains the Points and structure from the array
        """
        shape = array.shape
        solution = Solution(tuple(), shape[0])
        all_points = list()
        partitions = list()
        for partition in array:
            new_partition = Partition()
            for point in partition:
                new_point = Point(*point)
                new_partition.add(new_point)
                all_points.append(new_point)
            partitions.append(new_partition)
        solution.partitions = partitions
        solution.complete_graph = tuple(all_points)
        # solution.complete_graph = [Point(*point) for pat in array for point in pat]
        return solution

    def __init__(self, instance: Tuple[Point], partitions: int = 3, init_func=default_init,
                 old_solution=None):
        """
        Constructor of a new Solution.
        :param instance: List of all Points in the graph
        :param partitions: Number of Partitions for the solution
        :param init_func: initial mapping from points to partitions. If ist not specified the default_init
        :param old_solution: The from that the new Solution been derived
        """
        self.size = len(instance)
        self.partitions = [Partition() for x in range(partitions)]
        # TODO maybe as np.array 
        self.complete_graph = instance
        self.__old_solution = old_solution
        init_func(self)
        self.update_centers()

    def __len__(self):
        return self.size

    def update_centers(self):
        for part in self.partitions:
            part.update_center()

    def make_valid(self):
        for part in self.partitions:
            part.make_valid()

    def remove_empty_partiton(self):
        if not self.is_changed():
            return
        partitions_to_remove = list()
        # collect all empty partitions
        for part in self.partitions:
            if len(part) == 0:
                partitions_to_remove.append(part)
        # remove all empty partitions
        for part in partitions_to_remove:
            self.partitions.remove(part)

    def sort_partitions(self):
        """Sorts all Partitions in the solution in ascending order. So nearest Point to center is at index\
         0 and removed point is at last index. """
        for part in self.partitions:
            part.get_points().sort(key=lambda x: x - part.get_center())

    def get_centers(self) -> List[Point]:
        return [part.get_center() for part in self.partitions]

    def get_center_map_with_new(self) -> Dict[Point, Partition]:
        return {part.get_center(): Partition(part.get_center()) for part in self.partitions}

    def get_center_map(self) -> Dict[Point, Partition]:
        return {part.get_center(): part for part in self.partitions}

    def get_value(self) -> float:
        back = 0
        num_partitions = len(self.partitions)
        num_points = self.size

        for partition in self.partitions:
            points_in_partition = len(partition)
            sum_all_points = 0
            for point in partition:
                # TODO NUMPY parallelize
                sum_point = 0
                for other_point in partition:
                    sum_point += point - other_point
                sum_all_points += sum_point
            back += (points_in_partition * sum_all_points) / (num_points * num_partitions)
        return back

    def get_old_solution(self):
        return self.__old_solution

    def set_old_solution(self, origin: Solution):
        self.__old_solution = origin

    def is_changed(self) -> bool:
        for part in self.partitions:
            if part.is_changed():
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

    def to_BiPoint_list(self):
        back = list()
        for part in self.partitions:
            back.extend(part.to_BiPoint_list())
        return back

    def clone(self) -> Solution:
        """
        Creates a new Solution that is a independent copy of this object. Note that the complete graph is \
        not copied and is a reference to the single
        :return:
        """
        clone = copy(self)
        clone.partitions = deepcopy(self.partitions)
        return clone

    def new_with_self_as_old(self):
        """
        Creates a clone of the solution and set the Solution, that was called, as old_solution. This method is \
        equivalent to:
         new_solution = solution.clone()
         new_solution.set_old_solution(solution)
        :return:  New Solution where all is equivalent to the called object except the old_solution is the called object
        """
        back = self.clone()
        back.set_old_solution(self)
        return back


#### Partition


class Partition:

    # TODO reduce cals when updated/ not updated
    # distance onlyx update if changed

    def __init__(self, center: Point = Point()):
        """
        
        :param center: The center of the partition. Default ist the Origin
        """
        self.__invalid_expected_value = True
        self.__invalid_std_deviation = True
        self.__invalid_center = True
        self.__std_deviation = -1
        self._center = center
        self.__points_in_partition = list()
        self.__expected_value = -1

    def __abs__(self):
        return self.get_except_value()

    def __iter__(self):
        return self.__points_in_partition.__iter__()

    def __len__(self):
        return len(self.__points_in_partition)

    def __deepcopy__(self, memodict):
        """
        Copy all shallow. Only the points_in_partition is can be modified independent from others
        :param memodict: omitted
        :return: a copy of this object
        """
        back = copy(self)
        back.__points_in_partition = copy(back.__points_in_partition)
        return back

    def add(self, point: Point):
        """
        Add a Point to the Partition. Center and all other values are invalid
        :param point: Point to add
        :return: nothing
        """
        self.__points_in_partition.append(point)
        self.changed()

    def remove(self, point: Point):
        """
        Remove a Point
        :param point:
        :return:
        """
        try:
            self.__points_in_partition.remove(point)
        except ValueError:
            raise ValueError(f"Point: {point} is not in the Partition")
        self.changed()

    def update_center(self) -> bool:
        if not self.is_changed() or len(self) == 0:
            return True
        x_val = 0
        y_val = 0
        z_val = 0

        for point in self:
            x_val += point.x
            y_val += point.y
            z_val += point.z
        num_points = len(self)
        self._center = Point(x_val / num_points, y_val / num_points, z_val / num_points)
        self.__invalid_center = False
        return True

    def __update_expected_value(self):
        distance_sum = 0.0
        for p in self:
            # TODO NUMPY map + reduce
            distance_sum += p - self._center
        self.__expected_value = distance_sum / len(self)
        self.__invalid_expected_value = False

    def get_except_value(self) -> float:
        """
        is semantic equivalent to __abs__
        :return:
        """
        if self.__invalid_expected_value:
            self.__update_expected_value()
        return self.__expected_value

    def __update_standard_deviation(self):
        center = self.get_center()
        exp_value = self.get_except_value()
        update = 0
        for point in self:
            update += ((point - center) - exp_value) ** 2
        self.__std_deviation = update
        self.__invalid_std_deviation = False

    def get_standard_deviation(self) -> float:
        if self.__invalid_std_deviation:
            self.__update_standard_deviation()
        return self.__std_deviation

    def get_points(self) -> List[Point]:
        return self.__points_in_partition

    def set_points(self, points):
        self.__points_in_partition = points
        self.changed()

    def make_valid(self):
        # call all updates
        self.update_center()
        self.__update_standard_deviation()
        self.__update_expected_value()

    def changed(self):
        self.__invalid_center = True
        self.__invalid_std_deviation = True
        self.__invalid_expected_value = True

    def get_most_distant_point(self) -> Point:
        # Todo improve
        max_point = None
        max_dist = 0
        center = self.get_center()
        for p in self:
            dist = p - center
            if dist > max_dist:
                max_dist = dist
                max_point = p
        return max_point

    def get_as_three_lists(self):
        x = list()
        y = list()
        z = list()
        for point in self:
            x.append(point.x)
            y.append(point.y)
            z.append(point.z)
        return x, y, z

    def split_to_two(self) -> Tuple[Partition, Partition]:
        max_dist = self.get_most_distant_point()
        # Todo factor for new center depending on distance to max point
        new_center1 = max_dist / self._center
        vector = new_center1 // self._center
        new_center2 = self._center + vector

        new_part1 = Partition(new_center1)
        new_part2 = Partition(new_center2)
        for p in self:
            if p - new_center1 < p - new_center2:
                new_part1.add(p)
            else:
                new_part2.add(p)
        new_part2.update_center()
        new_part1.update_center()
        return new_part1, new_part2

    def get_center(self) -> Point:
        if self.__invalid_center:
            self.update_center()
        return self._center

    def is_changed(self) -> bool:
        return self.__invalid_center or self.__invalid_expected_value or self.__invalid_std_deviation

    def is_valid(self):
        return not self.is_changed()

    def to_numpy_array(self):
        complete = [point.to_tuple() for point in self]
        return np.array(complete)

    def to_BiPoint_list(self):
        return [BidirectPoint(*point.to_tuple(), partition=self) for point in self]
