from __future__ import annotations

import math

from typing import List

from core.Point import Point

import numpy as np

from core.Solution import Solution, Partition


def euclede_dist(point1: Point, point2: Point):
    return point1 - point2


class Action(object):
    pass


class Evaluator:

    def __init__(self):
        self.best = None

    def compare_and_update(self, *evaluations):
        top_five = [None] * 5
        for evaluation in evaluations:
            for index, pos in enumerate(top_five):
                if evaluation < pos:
                    continue
                else:
                    top_five[index] = evaluation
                    top_five.sort()
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
        self.anti_cost_func = lambda x : 1/x
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
        for point in partition.points_in_partition:
            shortest = math.inf
            min_point = None
            distance = 0
            # Todo edge case if only one point in partition
            for other_point in partition.points_in_partition:
                if point == other_point:
                    continue
                distance = point - other_point
                if distance < shortest:
                    shortest = distance
                    min_point = other_point
            eval_value += distance
            edges = np.concatenate(edges, np.array([point.to_tuple(), min_point.to_tuple(), shortest]))
        return eval_value
