from __future__ import annotations

import math
import random
from itertools import chain, combinations
from typing import Tuple, List

from core.Point import Point, random_Point
from core.Solution import Solution, Partition
from core.transformOperation import TransformationOperation  as tro, TransformationOperation

"""
def nearest_init(solution):
    solution.partitions

    pass
"""

"""
Level of operation- Concept:
    
    There are 3 levels of operation
     - Solution
     - Partition
     - Point
     
    Operations that assigned to Solution:
     - Add Partition
     - Remove Partition
     - Iterate Cluster
    
    Operations that assigned to Partition:
     - Split to two 
     - Reduce
     
    Operations that assigned to Point:
     - move_X_percent/ move_5_percent
     - move one specific point


"""


class Parameter(object):
    """
        possible parameters to check
        clusters
        points
        edges
    """

    pass

    def __init__(self):
        self.x = 0


# generate multiple solutions from current with different params neighbours
def transform_current_solution(run_solution, param_set):
    # run multiple transformations for current solution
    new_solutions = list()
    for params in param_set:
        # Todo parallelize
        # Note: copy needed
        new_solutions.append(transformation(run_solution, params))


def go_x_steps_back(solution: Solution, x: int):
    """
    To escape a stuck solution
    :param solution:
    :param x:
    :return: The Solution which was created x steps before plus the next used 
    """
    run = 0
    tmp_solution = solution
    child = None
    while run < x:
        if tmp_solution.get_old_solution() is not None:
            child = tmp_solution
            tmp_solution = tmp_solution.get_old_solution()
            run += 1
        else:
            raise Exception(f"The parameter X was too high there is/are only {run} steps done before")
    return tmp_solution, child


def transformation(run_solution: Solution) -> List[Solution]:
    """
    Decide between cluster_iteration, add_cluster, reduce_cluster, move_point, move_X_percent,  TODO AND ...
    :param run_solution:
    :param params:
    :return:
    """
    neighborhood = list()
    # iterate
    neighborhood.append(iterate_n_times(run_solution, 5))
    # add
    neighborhood.append((add_partition(run_solution, 5, random_Point())))
    # remove
    neighborhood.append(remove_partition(run_solution, 5))
    # move
    neighborhood.append(move_5_percent(run_solution))
    # moveX
    neighborhood.append(move_x_percent(run_solution, 10))
    # split
    neighborhood.append(split_cluster(run_solution))
    # reduce
    neighborhood.append(reduce_cluster(run_solution))

    return neighborhood

    # how big is the neighborhood

    # create all neighbors

def advanced_transformation(run_solution : Solution, options: List[Tuple[TransformationOperation, ]]):
    back = list()
    for option in options:
        if True:
            pass


def cluster_iteration(run_solution: Solution) -> Solution:
    """
    Creates a new empty Solution and assign the points to the partition that has nearest center

    Num partitions is equal run_solution.
    :param run_solution: Original Solution
    :return: the new Solution with new assigned points and updated centers
    """
    back = Solution.empty_solution()
    back.set_old_solution(run_solution)
    back.change_instance(run_solution.get_instance())
    center_map = run_solution.get_center_map_with_new()
    # TODO parallelize
    for point in run_solution.get_instance():
        min_dist = math.inf
        min_dist_center = None
        # get smallest
        for center in center_map.keys():
            new_dist = point - center
            if new_dist < min_dist:
                min_dist = new_dist
                min_dist_center = center
        # assign point to matching partition
        new_partition = center_map[min_dist_center]
        new_partition.add(point)
    back.partitions = list(center_map.values())
    back.update_centers()
    return back


def iterate_n_times(solution: Solution, n: int = 5):
    """
    Runs N times the cluster_iteration and omits the history of the intermediate steps. The Old_Solution is the solution that is given by the parameter
    :param solution: The solution, that has to be iterated n times
    :param n: Number of iterations
    :return: A new Solution where n iterations of the clusters are done. The old_solution is the the solution that was \
    given as parameter
    """
    iteration = solution
    # Optional TODO can be converted in a work with a copy and don't create a new Object every iteration
    for i in range(n):
        iteration = cluster_iteration(iteration)
    iteration.set_old_solution(solution)
    return iteration


def split_cluster(run_solution: Solution) -> Solution:
    """

    :param run_solution:
    :return:
    """
    back = run_solution.clone()
    back.set_old_solution(run_solution)
    max_var = 0
    pos = 0
    for index, part in enumerate(back.partitions):
        var = part.get_except_value()
        if var > max_var:
            max_var = var
            pos = index
    max_cluster = back.partitions[pos]
    cluster1, cluster2 = max_cluster.split_to_two()
    back.partitions.remove(max_cluster)
    back.partitions.append(cluster1)
    back.partitions.append(cluster2)
    back.set_create_operation(tro.SPLIT)
    return back


def add_partition(run_solution: Solution, iterations: int, center: Point = None) -> Solution:
    back = run_solution.new_with_self_as_old()
    if center is None:
        center = random_Point()
    new_Partition = Partition(center)
    back.partitions.append(new_Partition)
    back = iterate_n_times(back, iterations)
    back.set_old_solution(run_solution)
    return back


def remove_partition(run_solution: Solution, iterations: int, partition: Partition = None) -> Solution:
    """
    Remove one Partition from the given Solution. If no specific Partition is given the Partition with the highest deviation will be removed
    :param run_solution:
    :param iterations:
    :param partition:
    :return:
    """
    back = run_solution.new_with_self_as_old()
    back.remove_empty_partition()
    if partition is not None:
        back.partitions.remove(partition)
    else:
        partition_to_remove = None
        max_deviation = -1
        for part in back.partitions:
            dev = part.get_standard_deviation()
            if dev > max_deviation:
                max_deviation = dev
                partition_to_remove = part
        back.partitions.remove(partition_to_remove)
    back = iterate_n_times(back, iterations)
    back.set_old_solution(run_solution)
    return back


def reduce_cluster(run_solution: Solution) -> Solution:
    """
    Select two partitions with smallest (center-) distance and merge them together.
    I.e. The number of partitions is reduced by one
    :param run_solution:
    :return: A copy of run_solution with one partition less
    """
    if len(run_solution.partitions) == 1:
        raise Exception("There can't be less than one cluster!")
    back = run_solution.clone()
    back.set_old_solution(run_solution)
    min_dist = math.inf
    cluster_to_reduce = list()
    pairs = combinations(back.partitions, 2)
    for part in pairs:
        dist = part[0].get_center() - part[1].get_center()
        if dist < min_dist:
            min_dist = dist
            cluster_to_reduce = part

    new_part = Partition()
    new_part.set_points(list(chain.from_iterable(cluster_to_reduce)))
    new_part.changed()
    new_part.update_center()

    back.partitions.remove(cluster_to_reduce[0])
    back.partitions.remove(cluster_to_reduce[1])
    back.partitions.append(new_part)
    return back


def move_point(point: Point, origin: Partition, destination: Partition):
    if point not in origin:
        raise Exception("The point is not in the selected partition")
    origin.remove(point)
    destination.add(point)


def move_x_percent(run_solution: Solution, x: int) -> Solution:
    """

    :param run_solution:
    :param x:
    :return:
    """
    # for all partitions select x Percent
    back = run_solution.clone()
    back.set_old_solution(run_solution)
    back.sort_partitions()
    centers = back.get_center_map()
    for part in back.partitions:
        # select X percent removed points from center
        start = int(len(part) * (100 - x) / 100)
        removed_points = part.get_points()[start:]
        for point in removed_points:
            min_other = math.inf
            index_other = -1
            # calc nearest other cluster
            for center_point in centers.keys():
                if center_point is part.get_center():
                    continue
                if center_point - point < min_other:
                    min_other = center_point - point
                    index_other = center_point
            # swap to other cluster
            move_point(point, part, centers[index_other])
    # update middle of swapped
    back.update_centers()
    return back


def move_5_percent(run_solution: Solution) -> Solution:
    return move_x_percent(run_solution, 5)


def first_solution(instance: Tuple[Point], num_part: int = -1) -> Solution:
    """
    ONLY FOR FIRST ITERATION. Dont use this in the computation
    :param instance:
    :param num_part:
    :return: an initial Solution for the run
    """
    if num_part <= 1:
        raise Exception("The initial num_part is not valid ")
    back = random_solution(instance) if num_part == -1 else Solution(instance, num_part)
    back.update_centers()
    return back


def random_solution(instance: Tuple[Point], ) -> Solution:
    back = Solution(instance, random.randint(1, 10))
    return back
