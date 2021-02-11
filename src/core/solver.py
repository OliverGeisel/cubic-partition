from __future__ import annotations

import math
import random
from itertools import chain, combinations
from typing import Tuple

from core.Point import Point
from core.Solution import Solution, Partition

"""
def nearest_init(solution):
    solution.partitions

    pass
"""


# generate multiple solutions from current with different params neighbours
def transform_current_solution(run_solution, param_set):
    # run multiple transformations for current solution
    new_solutions = list()
    for params in param_set:
        # Todo parallelize
        # Note: copy needed
        new_solutions.append(transformation(run_solution, params))


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


def cluster_iteration(run_solution: Solution) -> Solution:
    """
    Creates a new empty Solution and assign the points to the partition that has nearest center

    Num partitions is equal run_solution.
    :param run_solution: Original Solution
    :return: the new Solution with new assigned points and updated centers
    """
    back = Solution.empty_solution()
    back.set_old_solution(run_solution)
    back.complete_graph = run_solution.complete_graph
    center_map = run_solution.get_center_map()
    # TODO parallelize
    for point in run_solution.complete_graph:
        min_dist = math.inf
        min_dist_point = None
        # get smallest 
        for center in center_map.keys():
            new_dist = point - center
            if new_dist < min_dist:
                min_dist = new_dist
                min_dist_point = center
        # assign point to matching cluster
        new_partition = center_map[min_dist_point]
        new_partition.add(point)
    back.partitions = list(center_map.values())
    back.update_centers()
    return back


def split_cluster(run_solution: Solution) -> Solution:
    """

    :param run_solution:
    :return:
    """
    back = run_solution.clone()
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


def transformation(run_solution: Solution, params: Parameter = Parameter()) -> Solution:
    """
    Decide between cluster_iteration, add_cluster, reduce_cluster, move_point, move_X_percent,  TODO AND ...
    :param run_solution:
    :param params:
    :return:
    """

    return cluster_iteration(run_solution)


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
