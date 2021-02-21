import random
from itertools import product, repeat, cycle
from pathlib import Path
from typing import List, Tuple
import ipyvolume as ipv
import sys

import matplotlib.pyplot as plot
import ipywidgets as widgets

import helper.converter as convert
from generator import Generator
from core import solver, Evaluate
from core.Point import Point
from core.solver import Solution, transformation
from helper.shortcuts import print_iterative


def generate_instance() -> Tuple[List[Point], Solution]:
    generator = Generator.SphereGenerator()
    return generator.create_point_Instance().get_instance_and_correct_solution()


def solve(instance: Tuple[Point]) -> Solution:
    iterations = 5  # simulated annealing parameter T
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
        new_solutions = transformation(run_solution)
        # reduce

        # evaluate
        scores = [-1] * len(new_solutions)
        for index, neigboor in enumerate(new_solutions):
            scores[index] = Evaluate.naive_imp(neigboor)
        print_iterative(scores)
        # update check condition
        # "update T"
        # get best result
        best_score = min(scores)
        best_neighbor = scores.index(best_score)

        # pass
    return run_solution


list_of_colors = ["#ff0000",  # red
                  "#00ff00",  # blue
                  "#0000ff",  # green
                  "#00ffff",  # cyan
                  "#ff00ff",  # magenta
                  # "#ffff00",  # yellow
                  # "#ffffff", # White
                  "#000000"  # black
                  ]

list_of_marker = ['o',
                  '^',
                  's']

list_of_IPYVmarker = ['arrow',
                      'box',
                      'diamond',
                      'sphere',
                      # 'point_2d',
                      # 'square_2d',
                      'triangle_2d',
                      'circle_2d']

combi_IPV = list(product(list_of_colors, list_of_IPYVmarker))
random.shuffle(combi_IPV)
combi_IPV = cycle(combi_IPV)


def to_3D_view(solution: Solution, correct_solution: Solution = None):
    extra_figures = list()
    # create correct solution
    if correct_solution is not None:
        figure_correct = ipv.figure("correct")
        ipv.pylab.xyzlim(-1, 11)
        for part in correct_solution.partitions:
            marker = random.choice(list_of_IPYVmarker)
            color = random.choice(list_of_colors)
            temp = combi_IPV.__next__()
            ipv.scatter(*convert.partition_to_IpyVolume(part), marker=temp[1], color=temp[0])
        extra_figures.append(figure_correct)
    figure_result = ipv.figure("result")
    container = ipv.gcc()
    ipv.current.container = widgets.HBox(container.children)
    ipv.current.containers["result"] = ipv.current.container

    # crate computed solution
    for part in solution.partitions:
        marker = random.choice(list_of_IPYVmarker)
        color = random.choice(list_of_colors)
        temp = combi_IPV.__next__()
        ipv.scatter(*convert.partition_to_IpyVolume(part), marker=temp[1], color=temp[0])

    ipv.pylab.xyzlim(0, 11)
    ipv.current.container.children = list(ipv.current.container.children) + extra_figures
    ipv.show()


def save_as_html(name, dir: str = None):
    path = Path() if dir is None else Path(dir)
    # must not be used
    # if not path.is_dir():
    #     raise Exception("The given directory is no directory")
    # Todo Check if name end with .html
    ipv.save(str(path) + name + ".html", makedirs=True, title="3D visual")


def complete(final_solution: Solution, correct_solution: Solution = None) -> None:
    """
    Will print result in 3D world
    :return: Noting
    """
    global combi_IPV
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
    to_3D_view(final_solution, correct_solution)
    # save in a separate file
    save_as_html("final")

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
