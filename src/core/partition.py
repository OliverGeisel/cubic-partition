from __future__ import annotations

from copy import copy
from itertools import combinations
from typing import List, Tuple

import numpy as np

from core.point import Point, BidirectPoint, BidiectPointEncode


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

    def get_silhouette(self):
        pass

    def get_points(self) -> List[Point]:
        return self.__points_in_partition

    def set_points(self, points: List[Point]):
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
        new_part2.make_valid()
        new_part1.make_valid()
        return new_part1, new_part2

    def get_center(self) -> Point:
        if self.__invalid_center:
            self.update_center()
        return self._center

    def is_changed(self) -> bool:
        return self.__invalid_center or self.__invalid_expected_value or self.__invalid_std_deviation

    def is_valid(self) -> bool:
        return not self.is_changed()

    def to_numpy_array(self):
        complete = [point.to_tuple() for point in self]
        return np.array(complete)

    def to_numpy_array_bidirect(self, partition_unumber=-1):
        temp = self.to_BiPointEncode_list(partition_unumber)
        return np.array([point.to_tuple() for point in temp])

    def to_BiPoint_list(self) -> List[BidirectPoint]:
        return [BidirectPoint(*point.to_tuple(),index=point.index, partition=self) for point in self]

    def to_BiPointEncode_list(self, part_num) -> List[BidiectPointEncode]:
        return [BidiectPointEncode(*point.to_tuple(), index=point.index, partition_number=part_num) for point in self]

    def to_BiPoint_list_with_distance_map(self, map):
        return [(BidirectPoint(*point.to_tuple(), partition=self), map) for point in self]


class Subpartition(Partition):

    def __init__(self):
        pass


class SubParts:
    pass


class PlanePartition(Partition):

    def __init__(self, origin=Point()):
        super().__init__(origin)
        self.normal_vector = None
        self.tension_vector1 = None
        self.tension_vector2 = None
        self.coordinates = None

    def update_normal_vector(self, origin=None):
        def normal_origin(p1: Point, p2: Point) -> Tuple[float, float, float]:
            return p1.vector_product(p2)

        def other_origin(p1, p2, origin) -> Tuple[float, float, float]:
            vecpoint1 = Point(*(p1 - origin))
            vecpoint2 = Point(*(p2 - origin))
            return vecpoint1.vector_product(vecpoint2)

        func = normal_origin if origin is None else other_origin
        for triple in combinations(self, 3):
            vec1 = func(triple[0], triple[1])
            vec2 = func(triple[0], triple[2])
            vec3 = func(triple[1], triple[2])

    def __distance_from_plane(self, point) -> float:
        normal = self.coordinates
        normal_sum = normal[0] * point.x + normal[1] * point.y + normal[2] * point.z - normal[3]
        dist_point = abs(point)
        return normal_sum / dist_point

    def get_normal_vector(self):
        return self.normal_vector

    def __update_standard_deviation(self):
        deviation = 0.0
        for point in self:
            deviation += self.__distance_from_plane(point)
        self.__std_deviation = deviation / len(self)
        self.__invalid_std_deviation = False

    def __update_expected_value(self):
        self.__invalid_expected_value = False
        return True

    def update_center(self) -> bool:
        self.__invalid_center = False
        return True
