class PointGeneratorConf:

    def __init__(self, amount: int):
        self.amount = amount

    @classmethod
    def default_Conf(cls):
        return PointGeneratorConf(1 << 8)
