from __future__ import annotations

import math
from itertools import combinations

from typing import List

from core.Point import Point, BidirectPoint
from core.Solution import Solution, Partition

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


def in_same_part(p1: BidirectPoint, p2: BidirectPoint) -> bool:
    return p1.get_partition() is p2.get_partition()


def cost_default(p1, p2, p3) -> float:
    back = 0.0
    p1 - p2 + p1 - p3 + p2 - p3
    return back


def cost_neg_default(p1, p2, p3) -> float:
    return min(p1 - p2, p1 - p3, p2 - p3)


def naive_imp(solution: Solution, cost=cost_default, cost_neg=cost_neg_default):
    sum = 0.0
    three_points = combinations(solution.to_BiPoint_list(), 3)
    for triple in three_points:
        p1 = triple[0]
        p2 = triple[1]
        p3 = triple[2]
        x1 = in_same_part(p1, p2)
        x2 = in_same_part(p1, p3)
        x3 = in_same_part(p2, p3)
        x_sum = x1 * x2 * x3
        sum += cost(p1, p2, p3) * x_sum + cost_neg(p1, p2, p3) * (1 - x_sum)
    return sum
