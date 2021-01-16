from __future__ import annotations

import math
import random

from core.Point import Point
from core.Solution import Solution, Partition


def nearest_init(solution):
    solution.partitions

    pass


# generate multiple solutions from current with different params neighbours
def transform_current_solution(run_sulution, param_set):
    # run multiple transformations for current solution
    new_solutions = list()
    for params in param_set:
        # Todo parallelize
        # Note: copy needed
        new_solutions.append(transformation(run_sulution, params))


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


def create_updated_solution(run_solution: Solution) -> Solution:
    back = Solution.empty_solution()
    center_points = {part.get_center(): Partition() for part in run_solution.partitions}
    # TODO parallelize
    for point in run_solution.complete_graph:
        min_dist = math.inf
        min_dist_point = None
        # get smallest 
        for center in center_points.keys():
            new_dist = point - center
            if new_dist < min_dist:
                min_dist = new_dist
                min_dist_point = center
        new_partition = center_points[min_dist_point]
        new_partition.add(point)
    back.partitions = list(center_points.values())
    back.complete_graph = run_solution.complete_graph
    back.update_centers()
    return back


def add_cluster(run_solution: Solution) -> Solution:
    """

    :param run_solution:
    :return:
    """
    back = run_solution.clone()
    max_var = 0
    pos = 0
    for part in range(len(back.partitions)):
        var = back.partitions[part].get_variance()
        if var > max_var:
            max_var = var
            pos = part
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
    for part in back.partitions:
        for other in back.partitions:
            if part == other:
                continue
            dist = part.get_center() - other.get_center()
            if dist < min_dist:
                min_dist = dist
                cluster_to_reduce = [part, other]

    new_part = Partition()
    new_part.points_in_partition = cluster_to_reduce[0].points_in_partition + cluster_to_reduce[1].points_in_partition
    new_part.update_center()

    back.partitions.remove(cluster_to_reduce[0])
    back.partitions.remove(cluster_to_reduce[1])
    back.partitions.append(new_part)
    return back


def move_x_percent(run_solution: Solution, x: int) -> Solution:
    # for all partitions select x Percent

    # select X perecent farest points from center

    # calc nearest other cluster
    # swap to other cluster
    # update midlle of swaped
    return run_solution


def move_5_percent(run_solution: Solution) -> Solution:
    return move_x_percent(run_solution, 5)


def move_point(point: Point, origin: Partition, destination: Partition):
    if point not in origin.points_in_partition:
        raise Exception("The point is not in the selected partition")

    origin.remove(point)
    destination.add(point)


def transformation(run_solution: Solution, params: Parameter = Parameter()) -> Solution:
    """
    decide between cluster_update, 
    :param run_solution:
    :param params:
    :return:
    """

    return create_updated_solution(run_solution)


def first_solution(instance, num_part: int = 1) -> Solution:
    back = random_solution(instance) if num_part == 1 else Solution(instance, num_part)
    return back


def random_solution(instance, ) -> Solution:
    back = Solution(instance, random.randint(1, 10))
    return back
