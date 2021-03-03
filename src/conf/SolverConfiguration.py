class SolverConfiguration:

    @classmethod
    def default(cls):
        return SolverConfiguration(False, 100, False, True)

    def __init__(self, multiple_start: bool, iterations: int, advanced_transformation: bool, second_step: bool):
        # for dbscan
        self.second_step = second_step
        self.radius = .2
        self.min_elements = 3
        # advanced search
        self.advanced_transformation = advanced_transformation
        # general configs
        self.iterations = iterations
        self.sub_iterations = 5
        self.multiple_start = multiple_start
        self.num_start_partitions = 2
