from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import List, Tuple, Dict, Any

import numpy as np

from core.point import Point, BidirectPoint, BidiectPointEncode, Bipoint_np
from core.partition import Partition, PlanePartition, DBPartition
from core.transformOperation import TransformationOperation as tro, TransformationOperation


def default_init(solution: Solution):
    for point in solution.get_instance():
        index = random.randint(0, len(solution.partitions) - 1)
        solution.partitions[index].add(point)


class Solution(ABC):

    @staticmethod
    @abstractmethod
    def empty_solution() -> Solution:
        pass

    def __init__(self, instance: Tuple[Point], old_solution: Solution):
        self.__create_operation = tro.NOT_SPECIFIC
        self.__complete_graph = instance
        self.partitions = None
        self.__old_solution = old_solution

    def __len__(self):
        return len(self.__complete_graph)

    def get_old_solution(self):
        return self.__old_solution

    def set_old_solution(self, origin: Solution):
        self.__old_solution = origin

    def get_instance(self):
        return self.__complete_graph

    def change_instance(self, new_graph, new_default: bool = False):
        self.__complete_graph = new_graph
        self.size = len(new_graph)
        for part in self.partitions:
            part.changed()
        if new_default:
            default_init(self)
            self.update_centers()

    def set_create_operation(self, operation: TransformationOperation):
        self.__create_operation = operation

    def get_create_operation(self):
        return self.__create_operation

    def update_centers(self):
        for part in self.partitions:
            part.update_center()

    def make_valid(self):
        for part in self.partitions:
            part.make_valid()

    def remove_empty_partition(self):
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

    def get_centers(self) -> List[Point]:
        return [part.get_center() for part in self.partitions]

    def get_center_map(self) -> Dict[Point, Partition]:
        return {part.get_center(): part for part in self.partitions}

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

    def to_unmpy_array_bidirect(self):
        tmp = [part.to_numpy_array_bidirect(number) for number, part in enumerate(self.partitions)]
        return np.array(tmp)

    def to_BiPoint_list(self) -> List[BidirectPoint]:
        back = list()
        for part in self.partitions:
            back.extend(part.to_BiPoint_list())
        return back

    def to_BiPoint_list_with_distance_map(self, map) -> List[Tuple[BidirectPoint, Any]]:
        back = list()
        for part in self.partitions:
            back.extend(part.to_BiPoint_list_with_distance_map(map))
        return back

    def to_BiPointEncode_list(self) -> List[BidiectPointEncode]:
        back = list()
        for number, part in enumerate(self.partitions):
            back.extend(part.to_BiPointEncode_list(number))
        return back

    @abstractmethod
    def clone(self) -> Solution:
        pass

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

    @abstractmethod
    def get_center_map_with_new_Partition(self) -> Dict[Point, Partition]:
        return {part.get_center(): Partition(part.get_center()) for part in self.partitions}

    @abstractmethod
    def sort_partitions(self):
        pass


class ConcreteSolution(Solution):

    @staticmethod
    def empty_solution() -> ConcreteSolution:
        """
        Creates an empty Solution wirh no Point and 0 Partitions
        :return: an new empty Solution
        """
        return ConcreteSolution(tuple(), 0)

    @staticmethod
    def solution_from_numpy_array(array: np.ndarray, make_valid=True) -> ConcreteSolution:
        """
        Parse a Solution from a numpy-array, if the shape is matching
        :param array: the array with
        :param make_valid: calculate all necessary values for partitions and solution
        :return: the Solution that contains the Points and structure from the array
        """
        shape = array.shape
        solution = ConcreteSolution(tuple(), shape[0])
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
        solution.change_instance(tuple(all_points))
        # solution.complete_graph = [Point(*point) for pat in array for point in pat]
        if make_valid:
            solution.make_valid()
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
        super().__init__(instance, old_solution)
        self.partitions = [Partition() for partition in range(partitions)]
        # TODO maybe as np.array
        init_func(self)
        self.update_centers()

    def sort_partitions(self):
        """Sorts all Partitions in the solution in ascending order. So nearest Point to center is at index\
         0 and removed point is at last index. """
        for part in self.partitions:
            part.get_points().sort(key=lambda x: x - part.get_center())

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

    def get_center_map_with_new_Partition(self) -> Dict[Point, Partition]:
        return {part.get_center(): Partition(part.get_center()) for part in self.partitions}

    def to_np_encode_list(self):
        return [Bipoint_np(*point.to_tuple()) for point in self.to_BiPointEncode_list()]

    def clone(self) -> ConcreteSolution:
        """
               Creates a new Solution that is a independent copy of this object. Note that the complete graph is \
               not copied and is a reference to the single
               :return:
               """
        clone = copy(self)
        clone.partitions = deepcopy(self.partitions)
        return clone


class DBSolution(Solution):

    def sort_partitions(self):
        pass

    @staticmethod
    def empty_solution() -> DBSolution:
        return DBSolution(tuple(), 0)

    def get_center_map_with_new_Partition(self) -> Dict[Point, Partition]:
        return {part.get_center(): DBPartition(part.get_center()) for part in self.partitions}

    def __init__(self, instance: Tuple[Point], partitions: int = 3, old_solution=None, radius=0.2, min_elements=3,
                 init_fiunc=default_init):
        super().__init__(instance, old_solution)
        self.partitions = list()
        for part in range(partitions):
            self.partitions.append(DBPartition(radius=radius, min_elements=min_elements))
        default_init(self)
        self.update_centers()

    def clone(self) -> DBSolution:
        """
               Creates a new Solution that is a independent copy of this object. Note that the complete graph is \
               not copied and is a reference to the single
               :return:
               """
        clone = copy(self)
        clone.partitions = deepcopy(self.partitions)
        return clone

    def link(self):
        for p in self.partitions:
            p.linking()

    def split_partitions(self):
        all_new_partitions = list()
        for part in self.partitions:
            all_new_partitions.extend(part.split())
        self.partitions = all_new_partitions


def default_init_plane(solution: PlaneSolution):
    origin = solution.origin
    radius = solution.get_radius()
    size = len(solution)

    def vec_with_origin(p1, p2) -> Tuple[float, float, float]:
        return p1.vector_product(p2)

    # initial init for partitions
    for index_part, part in enumerate(solution.partitions):
        repeat = True
        while repeat:
            index1 = random.randint(0, size - 1)
            index2 = random.randint(0, size - 1)
            while index1 == index2:
                index2 = random.randint(0, size - 1)
            index3 = random.randint(0, size - 1)
            while index1 == index3 or index3 == index2:
                index3 = random.randint(0, size - 1)

            point1 = solution.get_instance()[index1]
            point2 = solution.get_instance()[index2]
            point3 = solution.get_instance()[index3]
            normal_vec1 = vec_with_origin(point1, point2)
            normal_vec1 = Point(*normal_vec1).get_normalized_vector()
            normal_vec1 = np.array(normal_vec1) if normal_vec1[0] >= 0 else np.array(normal_vec1) * -1
            normal_vec2 = vec_with_origin(point1, point3)
            normal_vec2 = Point(*normal_vec2).get_normalized_vector()
            normal_vec2 = np.array(normal_vec2) if normal_vec2[0] >= 0 else np.array(normal_vec2) * -1
            normal_vec3 = vec_with_origin(point2, point3)
            normal_vec3 = Point(*normal_vec3).get_normalized_vector()
            normal_vec3 = np.array(normal_vec3) if normal_vec3[0] >= 0 else np.array(normal_vec3) * -1
            vectors = [normal_vec1, normal_vec2, normal_vec3]
            if (vectors[0] - vectors[1]).sum() < 0.1 and (vectors[0] - vectors[2]).sum() < 0.1:
                same_vector = False
                # ceck if is already in use
                for other_part in solution.partitions[:index_part]:
                    if (other_part.normal_vector - vectors[0]).sum() < 0.1:
                        same_vector = True
                        break
                if same_vector:
                    continue
                # assign
                part.tension_vector1 = np.array(point1.get_normalized_vector())
                part.tension_vector2 = np.array(point2.get_normalized_vector())
                part.normal_vector = np.array(normal_vec1)
                repeat = False
    # assign points to partitions
    for point in solution.get_instance():
        if point - origin < radius:
            # point is too close to the origin
            solution.add_to_not_assigned(point)
        else:
            normal_vectors = solution.get_normal_and_tension_vectors()
            variances = dict()
            for index, vector in enumerate(normal_vectors):
                test_normal = point.vector_product(Point(*vector[1]))
                test_normal = np.array(Point(*test_normal).get_normalized_vector())
                test_normal = test_normal if test_normal[0] >= 0 else test_normal*-1
                distance = np.abs(vector[0] - test_normal).sum()
                variances[distance] = index
            key = min(variances.keys())
            index = variances[key]
            solution.partitions[index].add(point)
    return solution


class PlaneSolution(Solution):

    @staticmethod
    def empty_solution() -> PlaneSolution:
        """
        Creates an empty Solution wirh no Point and 0 Partitions
        :return: an new empty Solution
        """
        return PlaneSolution(tuple(), 0)

    def get_center_map_with_new_Partition(self) -> Dict[Point, Partition]:
        return {part.get_center(): PlanePartition(part.get_center()) for part in self.partitions}

    def clone(self) -> PlaneSolution:
        clone = copy(self)
        clone.partitions = deepcopy(self.partitions)
        return clone

    def __init__(self, instance: Tuple[Point], partitions: int, init_func=default_init_plane,
                 old_solution: ConcreteSolution = None,
                 origin=Point(), radius=2):
        super().__init__(instance, old_solution)
        self.partitions = [PlanePartition(origin) for partition in range(partitions)]
        # TODO maybe as np.array
        self.origin = origin
        self.__radius = radius
        self.__not_assigned_points = list()
        init_func(self)
        self.update_centers()

    def get_centers(self) -> List[Point]:
        raise Exception("The centers are not available in PlaneSolutions ")

    def get_radius(self):
        return self.__radius

    def set_radius(self, value):
        self.__radius = value

    def add_to_not_assigned(self, point: Point):
        self.__not_assigned_points.append(point)

    def get_not_assigned_points(self) -> List[Point]:
        return self.__not_assigned_points

    def get_normal_vectors(self):
        return [part.get_normal_vector() for part in self.partitions]

    def get_normal_and_tension_vectors(self):
        return [part.get_normal_and_tension_vectors() for part in self.partitions]

    def not_assigned_points_to_partition(self):
        new_part = Partition()  # to mark its not normal
        for point in self.__not_assigned_points:
            new_part.add(point)
        self.partitions.append(new_part)
        self.__not_assigned_points.clear()

    def not_assigned_back_to_list(self):
        # Todo Implement
        print("Sorry not implemented yet!")
        pass

    def assign_all(self, points: List[Point]):
        for point in points:
            self.__assign_to_best_part(point)

    def __assign_to_best_part(self, point: Point):
        best_part = None
        min_variance = math.inf
        for index, vector in enumerate(self.get_normal_and_tension_vectors()):
            normal_vector_abs = math.sqrt(np.square(vector[0]).sum())
            normal_vector = vector[0] / normal_vector_abs
            normal_vector = normal_vector if normal_vector[0] >= 0 else normal_vector * -1
            new_normal = point.vector_product(Point(*vector[1]))
            new_normal = Point(*new_normal).get_normalized_vector()
            new_normal = np.array(new_normal) if new_normal[0] >= 0 else np.array(new_normal) * -1
            if np.abs(new_normal - normal_vector).sum() < min_variance:
                best_part = self.partitions[index]
        best_part.add(point)

    def sort_partitions(self):
        """Sorts all Partitions in the solution in ascending order. So nearest Point to center is at index\
         0 and removed point is at last index. """
        for part in self.partitions:
            def sort_func(x: Point):
                normalvector = Point(*x.vector_product(part.tension_vector1)).get_normalized_vector()
                normalvector = np.array(normalvector) if normalvector[0] >= 0 else np.array(normalvector) * -1
                return np.abs(normalvector - part.normal_vector).sum()

            part.get_points().sort(key=sort_func)

    def complete(self):
        for unassigned_point in self.__not_assigned_points:
            self.__assign_to_best_part(unassigned_point)
        self.__not_assigned_points.clear()

    def reduce_radius(self, reduce: float):
        """
        Rduce the radius of the forbidden zone by the value of reduce. and assign the new point to the best fitting partition
        :param reduce: amount how to decrease the radius
        :return: None
        """
        if reduce < 0.0 or reduce > self.__radius:
            raise Exception(f"Reduce is to large! Plese use a value that is smaller than the radius: {self.__radius}")
        self.__radius -= reduce
        # update not assigned points
        point_to_assign = list()
        for point in self.__not_assigned_points:
            if abs(point) > self.__radius:
                point_to_assign.append(point)
        for point in point_to_assign:
            self.__not_assigned_points.remove(point)
        for point in point_to_assign:
            self.__assign_to_best_part(point)
