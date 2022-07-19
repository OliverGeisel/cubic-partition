import time
from multiprocessing import Pool, Process, shared_memory
from typing import Tuple, List

import numpy as np

from core import solver, evaluate
from core.point import Point
from core.solution import ConcreteSolution
from generator.generator import SphereGenerator
from main import complete


def generate_instance() -> Tuple[List[Point], ConcreteSolution]:
    generator = SphereGenerator()
    return generator.create_point_Instance().get_instance_and_correct_solution()


def f(x):
    back = x + x + x + x
    time.sleep(.2)
    shm = shared_memory.SharedMemory(name="test")
    mem = np.ndarray((10000, 10), dtype=np.float32, buffer=shm.buf)
    print(mem[0])
    print(back)
    return back


def set_global_indexes(solution):
    count = 0
    for point in solution.get_instance():
        point.index = count
        count += 1


if __name__ == "__main__":
    instance, correct_colution = generate_instance()
    run_solution = solver.first_solution(instance, 2)
    # iterate initial
    # calc all distances
    set_global_indexes(run_solution)
    all_dist = list()
    for p1 in run_solution.get_instance():
        new_line = list()
        for p2 in run_solution.get_instance():
            new_line.append(p1 - p2)
        all_dist.append(new_line)

    distance_map = np.array(all_dist, dtype=np.float32)
    shm = shared_memory.SharedMemory(create=True, size=distance_map.nbytes, name="distance_map")
    tmp = np.ndarray(distance_map.shape, dtype=distance_map.dtype, buffer=shm.buf)
    tmp[:] = distance_map[:]  # Copy the original data into shared memory
    # iterate initial
    run_solution = solver.iterate_n_times(run_solution, 10)
    start = time.perf_counter()
    best_score = evaluate.naive_imp_fast(run_solution, False)
    end = time.perf_counter()
    print(f"Time init: {end - start}")
    shm.unlink()
    complete(run_solution)
    # p = Process(target=f, args=(5,))
    # print("okay")
    # p.start()
    # p.join()
    # values =[ [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]] *10000
    # array = np.array(values,dtype=np.float32)
    # shm = shared_memory.SharedMemory(name="test", create=True,size=array.nbytes)
    # sh_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    # sh_array[:] = array[:]
    # with Pool() as p:
    #     p.map(f, values)
    # for x in values:
    #     print(x)
