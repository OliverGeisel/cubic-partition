class SolverConfiguration:

    @classmethod
    def default(cls):
        return SolverConfiguration(False, 100, False, True)

    def __init__(self, multiple_start: bool, iterations: int, advanced_transformation: bool, second_step: bool):
        self.advanced_transformation = advanced_transformation
        self.iterations = iterations
        self.sub_iterations = 5
        self.multiple_start = multiple_start
        self.second_step = second_step
