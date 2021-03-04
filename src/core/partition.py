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
        self._invalid_expected_value = True
        self._invalid_std_deviation = True
        self._invalid_center = True
        self.__std_deviation = -1
        self._center = center
        self.__points_in_partition = list()
        self._expected_value = -1

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
        self._invalid_center = False
        return True

    def __update_expected_value(self):
        distance_sum = 0.0
        for p in self:
            # TODO NUMPY map + reduce
            distance_sum += p - self._center
        self._expected_value = distance_sum
        if len(self) != 0:
            self._expected_value /= len(self)
        self._invalid_expected_value = False

    def get_except_value(self) -> float:
        """
        is semantic equivalent to __abs__
        :return:
        """
        if self._invalid_expected_value:
            self.__update_expected_value()
        return self._expected_value

    def __update_standard_deviation(self):
        center = self.get_center()
        exp_value = self.get_except_value()
        update = 0
        for point in self:
            update += ((point - center) - exp_value) ** 2
        self.__std_deviation = update
        self._invalid_std_deviation = False

    def get_standard_deviation(self) -> float:
        if self._invalid_std_deviation:
            self.__update_standard_deviation()
        return self.__std_deviation

    def get_silhouette(self):
        pass

    def get_points(self) -> List[Point]:
        return self.__points_in_partition

    def set_points(self, points: List[Point]):
        self.__points_in_partition = [point for point in points]
        self.changed()

    def make_valid(self):
        # call all updates
        self.update_center()
        self.__update_standard_deviation()
        self.__update_expected_value()

    def changed(self):
        self._invalid_center = True
        self._invalid_std_deviation = True
        self._invalid_expected_value = True

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
        if self._invalid_center:
            self.update_center()
        return self._center

    def is_changed(self) -> bool:
        return self._invalid_center or self._invalid_expected_value or self._invalid_std_deviation

    def is_valid(self) -> bool:
        return not self.is_changed()

    def to_numpy_array(self):
        complete = [point.to_tuple() for point in self]
        return np.array(complete)

    def to_numpy_array_bidirect(self, partition_number=-1):
        temp = self.to_BiPointEncode_list(partition_number)
        return np.array([point.to_tuple() for point in temp])

    def to_BiPoint_list(self) -> List[BidirectPoint]:
        return [BidirectPoint(*point.to_tuple(), index=point.index, partition=self) for point in self]

    def to_BiPointEncode_list(self, part_num) -> List[BidiectPointEncode]:
        return [BidiectPointEncode(*point.to_tuple(), index=point.index, partition_number=part_num) for point in self]

    def to_BiPoint_list_with_distance_map(self, map):
        return [(BidirectPoint(*point.to_tuple(), partition=self), map) for point in self]


class Subpartition(Partition):

    def __init__(self):
        pass


class DBPartition(Partition):

    def __init__(self, center: Point = Point(), radius: float = .2, min_elements: int = 3):
        super().__init__(center)
        self.noise_points = list()
        self.core_points = list()
        self.edge_points = list()
        self.radius = radius
        self.min_elements = min_elements
        self.subpartitions = None

    def linking(self):
        link_map = np.zeros(shape=(len(self.core_points), len(self.core_points)))
        for current_point_index, core_point in enumerate(self.core_points):
            for index2, second_core in enumerate(self.core_points):
                if core_point - second_core < self.radius:
                    link_map[current_point_index, index2] = 1
        linked_points = set()

        sub_parts = list()
        list_of_cores = copy(self.core_points)
        last_point = 0
        while len(list_of_cores) > 0 and last_point < len(self.core_points):
            queque = list()
            queque.append((list_of_cores[0], last_point))
            changed = True
            while changed:
                changed = False
                point, current_point_index = queque.pop(0)
                linked_points.add(point)
                list_of_cores.remove(point)
                # find all linked core points // same point not included
                for other_point_index, other_point in enumerate(
                        link_map[current_point_index][current_point_index + 1:]):
                    if other_point == 1:  # if close enough together
                        last_point = current_point_index + other_point_index + 1
                        queque.append((self.core_points[last_point], last_point))
                        changed = True
            # add 1 to last point for next core point who is not connected to the rest
            last_point += 1
            sub_parts.append(linked_points)
        assign_edge_points = {}
        for point in self.edge_points:
            restart = False
            for sub_part_index, sub_part in enumerate(sub_parts):
                for center in sub_part:
                    if point - center < self.radius:
                        assign_edge_points[point] = sub_part_index
                        restart = True
                        break
                if restart:
                    break
        for point, index in assign_edge_points.items():
            sub_parts[index].add(point)
        self.subpartitions = sub_parts

    def split(self) -> List[Partition]:
        """
        Splits the partition to the new sub partitons depending on the dbscan
        :return:
        """
        self.linking()
        back = list()
        for part in self.subpartitions:
            new_part = Partition()
            for point in part:
                new_part.add(point)
            back.append(new_part)
        return back


class PlanePartition(Partition):

    def __init__(self, origin=Point()):
        super().__init__(origin)
        self.normal_vector = None
        self.tension_vector1 = None
        self.tension_vector2 = None
        self.coordinates = None

    def update_coordinates(self):
        new_coords = np.zeros(4, dtype=np.float32)
        new_coords[:3] = self.normal_vector[:]
        new_coords[3] = np.sum(new_coords + np.array(self.get_center()))

    def update_normal_vector(self, origin=None):
        def normal_origin(p1: Point, p2: Point) -> Tuple[float, float, float]:
            return p1.vector_product(p2)

        def other_origin(p1, p2, origin) -> Tuple[float, float, float]:
            vecpoint1 = Point(*(p1 - origin))
            vecpoint2 = Point(*(p2 - origin))
            return vecpoint1.vector_product(vecpoint2)

        new_normal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        func = normal_origin if origin is None else other_origin
        triples = combinations(self, 3)
        size = 0
        for triple in triples:
            vec1 = np.array(func(triple[0], triple[1]), dtype=np.float32)
            vec2 = np.array(func(triple[0], triple[2]), dtype=np.float32)
            vec3 = np.array(func(triple[1], triple[2]), dtype=np.float32)
            # set all vec to same direction
            new_normal += vec1 * -1 if vec1[0] < 0 else vec1
            new_normal += vec2 * -1 if vec2[0] < 0 else vec2
            new_normal += vec3 * -1 if vec3[0] < 0 else vec3
            size += 3
        if size != 0:
            new_normal /= size
        new_vec_abs = np.sqrt(np.square(new_normal).sum())
        self.normal_vector = new_normal / (new_vec_abs if new_vec_abs > 0 else 1)

    def __derivation_to_normal_vector_and_t1(self, point: Point):
        new_vector = point.vector_product(Point(*self.tension_vector1))
        new_vector = Point(*new_vector).get_normalized_vector()
        new_vector = np.array(new_vector) if new_vector[0] >= 1 else np.array(new_vector) * -1
        return np.abs(new_vector - self.normal_vector).sum()

    def get_most_distant_point(self) -> Point:
        # Todo improve
        max_point = None
        max_dist = 0
        for p in self:
            dist = self.__derivation_to_normal_vector_and_t1(p)
            if dist > max_dist:
                max_dist = dist
                max_point = p
        return max_point

    def split_to_two(self) -> Tuple[Partition, Partition]:
        # Todo factor for new center depending on distance to max point

        new_part1 = PlanePartition()
        new_part2 = PlanePartition()
        std_deviation = self.get_standard_deviation()
        for p in self:
            if self.__derivation_to_normal_vector_and_t1(p) < std_deviation:
                new_part1.add(p)
            else:
                new_part2.add(p)
        new_part2.make_valid()
        new_part1.make_valid()
        return new_part1, new_part2

    def __distance_from_plane(self, point) -> float:
        normal = self.coordinates
        normal_sum = normal[0] * point.x + normal[1] * point.y + normal[2] * point.z - normal[3]
        dist_point = abs(point)
        return normal_sum / dist_point

    def get_normal_vector(self):
        return self.normal_vector

    def get_normal_and_tension_vectors(self):
        return self.normal_vector, self.tension_vector1, self.tension_vector2

    def __update_standard_deviation(self):
        deviation = 0.0
        for point in self:
            deviation += self.__distance_from_plane(point)
        self.__std_deviation = deviation / len(self)
        self.__invalid_std_deviation = False

    def __update_expected_value(self):
        update = 0.0
        for point in self:
            update += self.__distance_from_plane(point)
        self._expected_value = update
        self.__invalid_expected_value = False
        return True

    def update_center(self) -> bool:
        self.update_normal_vector()
        self._invalid_center = False
        return True
