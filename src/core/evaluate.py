from __future__ import annotations

import math
from itertools import combinations
from multiprocessing import Pool, shared_memory

from typing import List

from core.point import Point, BidirectPoint
from core.solution import ConcreteSolution, Partition, PlaneSolution, Solution

import numpy as np


def euclede_dist(point1: Point, point2: Point):
    return point1 - point2


class Action(object):
    pass


class Evaluator:

    def __init__(self):
        self.best = None

    def compare_and_update(self, *evaluations: Evaluation):
        top_five = [None] * 5
        for evaluation in evaluations:
            for index, pos in enumerate(top_five):
                if evaluation < pos:
                    top_five.insert(index, evaluation)
                    top_five.pop(index + 1)
                    break
        potential_best = top_five[0]
        if potential_best < self.best:
            self.best = potential_best

    def create_possible_action_of_best(self) -> List[Action]:
        # this is the magic here we can decide what new Solutions should be produced

        pass


class Evaluation:

    def __init__(self, solution: Solution):
        self.solution = solution
        self.best = None
        self.max_back = 5
        self.cost_func = lambda x: x
        self.anti_cost_func = lambda x: 1 / x
        self.possible_options = None

    def __lt__(self, other):
        if other is None:
            return True
        return self.solution.get_value() < other.solution.get_value()

    @staticmethod
    def create_evaluation(solution) -> Evaluation:
        back = Evaluation(solution)
        return back

    def eval(self):

        # eva_func ist evaluation function to
        eva_func = None
        sum_of_partition_abs = 0
        for partition in self.solution.partitions:
            sum_of_partition_abs += abs(partition)

        self.compare(self.solution.get_value())

    def compare(self, eval_instance):
        back = eval_instance < self.best
        if back:
            self.best = eval_instance
        return back

    def eval_Point_distance(self, partition: Partition) -> float:
        edges = np.array(0, dtype=np.float32)
        eval_value = 0
        for point in partition:
            shortest = math.inf
            min_point = None
            distance = 0
            # Todo edge case if only one point in partition
            for other_point in partition:
                if point == other_point:
                    continue
                distance = point - other_point
                if distance < shortest:
                    shortest = distance
                    min_point = other_point
            eval_value += distance
            edges = np.concatenate(edges, np.array([point.to_tuple(), min_point.to_tuple(), shortest]))
        return eval_value


def partition_silhouette(solution) -> List[float]:
    pass


def in_same_part(p1: BidirectPoint, p2: BidirectPoint) -> bool:
    return p1._partition is p2._partition


def cost_default(p1, p2, p3) -> float:
    back = (p1 - p2) + (p1 - p3) + (p2 - p3)
    return back


def cost_neg_default(p1, p2, p3) -> float:
    return min(p1 - p2, p1 - p3, p2 - p3) * .01


def in_same_part2(p1, p2) -> bool:
    return p1[3] == p2[3]


def cost_default2(p1, p2, p3) -> float:
    back = 0.0
    d1 = p1 - p2
    d2 = p1 - p3
    d3 = p2 - p3
    back = math.sqrt(np.square(d1[:-1]).sum()) + math.sqrt(np.square(d2[:-1]).sum()) + math.sqrt(
        np.square(d3[:-1]).sum())
    return back


def cost_neg_default2(p1, p2, p3) -> float:
    d1 = p1 - p2
    d2 = p1 - p3
    d3 = p2 - p3
    p12 = math.sqrt(np.square(d1[:-1]).sum())
    p13 = math.sqrt(np.square(d2[:-1]).sum())
    p23 = math.sqrt(np.square(d3[:-1]).sum())
    return min(p12, p23, p13)


def naive_imp(solution: Solution, cost=cost_default, cost_neg=cost_neg_default):
    sum = 0.0
    three_points = combinations(solution.to_BiPoint_list(), 3)
    for triple in three_points:
        p1 = triple[0]
        p2 = triple[1]
        p3 = triple[2]
        x1 = p1._partition is p2._partition
        x2 = p1._partition is p3._partition
        x3 = p2._partition is p3._partition
        x_sum = x1 * x2 * x3

        sum += cost(p1, p2, p3) if x_sum else cost_neg(p1, p2, p3)
    sum /= len(solution)
    sum *= len(solution.partitions)
    return sum


def calc(triple):
    pass


def calc_p(triple):
    p1 = triple[0]
    p2 = triple[1]
    p3 = triple[2]
    dist_map = shared_memory.SharedMemory(name="distance_map")
    size = int(math.sqrt(len(dist_map.buf)))
    x1 = p1._partition is p2._partition
    x2 = p1._partition is p3._partition
    x3 = p2._partition is p3._partition
    x_sum = x1 * x2 * x3
    if dist_map is None:
        dist1 = p1 - p2
        dist2 = p1 - p3
        dist3 = p2 - p3
    else:
        dist1 = dist_map.buf[p1.index * size + p2.index]
        dist2 = dist_map.buf[p1.index * size + p3.index]
        dist3 = dist_map.buf[p2.index * size + p3.index]
    dist_map.close()
    return (dist1 + dist2 + dist3) if x_sum else min(dist1, dist2, dist3) * .01


def naive_imp_fast(solution: Solution, parallel=False):
    result = 0.0
    # calc point penalty
    three_points = combinations(solution.to_BiPoint_list(), 3)
    # first version of parallel process
    shm = shared_memory.SharedMemory(name="distance_map")
    distance_map = np.ndarray((solution.size, solution.size), dtype=np.float32, buffer=shm.buf)
    # three_points = [(*triple, distance_map.buf) for triple in three_points]
    # TODO find race condition and solve dist_map
    if parallel:
        with Pool(maxtasksperchild=100) as pool:
            results = pool.imap(calc_p, three_points, chunksize=solution.size)
        result = sum(results)
    else:
        for triple in three_points:
            p1 = triple[0]
            p2 = triple[1]
            p3 = triple[2]

            x1 = p1._partition is p2._partition
            x2 = p1._partition is p3._partition
            x3 = p2._partition is p3._partition
            x_sum = x1 * x2 * x3
            if distance_map is None:
                dist1 = p1 - p2
                dist2 = p1 - p3
                dist3 = p2 - p3
            else:
                dist1 = distance_map[p1.index][p2.index]
                dist2 = distance_map[p1.index][p3.index]
                dist3 = distance_map[p2.index][p3.index]
            result+= (dist1 + dist2 + dist3) if x_sum else min(dist1, dist2, dist3) * .01

        #result = sum(map(calc, three_points))
    # add += (dist1 + dist2 + dist3) if x_sum else min(dist1, dist2, dist3) * .01
    shm.close()
    result /= len(solution)
    result *= len(solution.partitions)
    # calc partition penalty
    num_parts = len(solution.partitions)
    partition_penalty = 0
    for part in solution.partitions:
        size = len(part)
        min_distance = math.inf
        other_size = 0
        center_map = solution.get_center_map()
        for center in center_map.keys():
            if center is part.get_center():
                continue
            dist = center - part.get_center()
            if min_distance > dist:
                min_distance = dist
                other_size = len(center_map[center])
        partition_penalty += (size + other_size) / (min_distance * 2) * (
                result / len(solution)) * 0.5  # factor depending of variation and other values of partitions
    return result + partition_penalty


# for np arrays
def naive_imp2(solution: Solution, cost=cost_default2, cost_neg=cost_neg_default2):
    sum = 0.0
    tmp1 = [point.to_tuple() for point in solution.to_BiPointEncode_list()]
    three_points = combinations(np.array(tmp1), 3)
    for triple in three_points:
        p1 = triple[0]
        p2 = triple[1]
        p3 = triple[2]
        x1 = in_same_part2(p1, p2)
        x2 = in_same_part2(p1, p3)
        x3 = in_same_part2(p2, p3)
        x_sum = x1 * x2 * x3
        sum += cost(p1, p2, p3) * x_sum + cost_neg(p1, p2, p3) * (1 - x_sum)
    sum /= len(solution)
    sum *= len(solution.partitions)

    return sum


def default_plane_cost(point1, point2, point3):
    pass


def default_plane_negative_cost(point1, point2, point3):
    pass


def naive_plane_imp(solution: PlaneSolution):
    """
    Only use if origin == (0,0,0)
    :param solution:
    :return:
    """
    all_points = solution.to_BiPoint_list()
    all_triples = combinations(all_points, 3)
    origin = Point()
    eval_value = 0.0
    # TODO parallels
    for triple in all_triples:
        p1 = triple[0]
        p2 = triple[1]
        p3 = triple[2]
        # get normal vectors and normalize them 
        vector1 = Point(*p1.vector_product(p2)).get_normalized_point()
        vector2 = Point(*p1.vector_product(p3)).get_normalized_point()
        vector3 = Point(*p2.vector_product(p3)).get_normalized_point()
        # compare
        result1 = vector1 == vector2 or vector1 == (vector2 * -1)
        result2 = vector1 == vector3 or vector1 == (vector3 * -1)
        result3 = vector2 == vector3 or vector2 == (vector3 * -1)
        result_sum = result1 * result2 * result3
        eval_value += default_plane_cost(p1, p2, p3) if result_sum else default_plane_negative_cost(p1, p2, p3)
