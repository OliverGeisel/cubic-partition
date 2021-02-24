from typing import List, Tuple

from core.point import Point


class PlaneGeneratorConf(object):

    def __init__(self, points_per_plane: int, planes: int, max_spread: float = 0.0,
                 divergence: Tuple[float, float] = (-5, 5),
                 *origins: Tuple[Point, Point, Point]):
        self.points_per_cluster = points_per_plane
        self.divergence = divergence
        self.planes = planes
        self.max_spread = max_spread
        self.origins = origins

    @staticmethod
    def default_Conf():
        random_planes_from_origin = (Point(), None, None)
        return PlaneGeneratorConf(1 << 8, 2, 0.0, (-10, 10), random_planes_from_origin)
