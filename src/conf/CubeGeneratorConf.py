class CubeGeneratorConf(object):

    def __init__(self, points_per_cluster: int, clusters: int, max_spread: float):
        self.points_per_cluster = points_per_cluster
        self.clusters = clusters
        self.max_spread = max_spread

    @staticmethod
    def default_Conf():
        return CubeGeneratorConf(1 << 8, 5, 0.5)

    pass
