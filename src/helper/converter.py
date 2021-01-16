import numpy as np

from core.solver import Partition


def partition_to_IpyVolume(partition: Partition):
    x, y, z = partition.get_as_three_lists()
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    return x, y, z
