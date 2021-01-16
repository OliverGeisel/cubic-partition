import random
from typing import List, Tuple
import ipyvolume as ipv
import numpy as np
import sys

import matplotlib.pyplot as plot
import ipywidgets as widgets

import helper.converter as convert
from generator import Generator
from core import solver, Evaluate
from core.Point import Point
from core.solver import Solution, transformation


def generate_instance() -> Tuple[List[Point], Solution]:
    generator = Generator.SphereGenerator()
    return generator.create_point_Instance(200, 2, 2).get_instance_and_correct_solution()


def solve(instance) -> Solution:
    iterations = 5
    check_condition = True
    evaluator = Evaluate.Evaluation(None)
    # create random solution
    run_solution = solver.first_solution(instance, 2)
    # evaluate
    # evaluated_value = evaluator.eval(run_solution)
    for x in range(iterations):
        # iterate to get a best solution (local minima)
        # Todo need config how often long and exact
        # call solver functions
        run_solution = transformation(run_solution)
        # reduce

        # evaluate

        # update check condition
        # pass
    return run_solution


list_of_colors = ["#ff0000",  # red
                  "#00ff00",  # blue
                  "#0000ff",  # green
                  "#00ffff",  # cyan
                  "#ff00ff",  # yellow
                  "#ffff00",  # magenta
                  # "#ffffff", # White
                  "#000000"   # black
                  ]

list_of_marker = ['o',
                  '^',
                  's']

list_of_IPYVmarker = ['diamond',
                      'arrow',
                      'box',
                      'sphere']


def complete(final_solution: Solution, correct_solution: Solution = None) -> None:
    """
    Will print result in 3D world
    :return: Noting
    """

    # init of plot-output #1
    figure = plot.figure()
    axes = figure.add_subplot(111, projection='3d')

    for part in final_solution.partitions:
        color = random.choice(list_of_colors)
        marker = random.choice(list_of_marker)
        for p in part:
            axes.scatter(p.x, p.y, p.z, marker=marker, c=color)

    axes.set_xlabel("X Achse")
    axes.set_ylabel("Y Achse")
    axes.set_zlabel("Z Achse")
    plot.show()

    # init 3D view
    ipv.current.containers.clear()
    ipv.current.figures.clear()

    extra_figures = list()
    # create correct solution
    if correct_solution is not None:
        figure_correct = ipv.figure("correct")
        ipv.pylab.xyzlim(-1, 11)
        for part in correct_solution.partitions:
            marker = random.choice(list_of_IPYVmarker)
            color = random.choice(list_of_colors)
            ipv.scatter(*convert.partition_to_IpyVolume(part), marker=marker, color=color)
        extra_figures.append(figure_correct)
    figure_result = ipv.figure("result")
    container = ipv.gcc()
    ipv.current.container = widgets.HBox(container.children)
    ipv.current.containers["result"] = ipv.current.container

    # crate computed solution
    for part in final_solution.partitions:
        marker = random.choice(list_of_IPYVmarker)
        color = random.choice(list_of_colors)
        ipv.scatter(*convert.partition_to_IpyVolume(part), marker=marker, color=color)

    ipv.pylab.xyzlim(-1, 11)
    ipv.current.container.children = list(ipv.current.container.children) + extra_figures
    ipv.show()
    # save in a separate file
    ipv.save("example.html")

    # destroy 3D views
    ipv.clear()


def run():
    print("hello")
    # generate instance  set of 3D points
    instance, correct_colution = generate_instance()
    # find best solution for cubic partition problem
    solution = solve(instance)
    # print result
    complete(solution, correct_colution)


if __name__ == "__main__":
    args = sys.argv
    for arg in args:
        print(arg)
    run()
