from typing import List
import json

class mainConfig:

    #

    def __init__(self, initial: List = None, cuda: bool = False, notebook: bool = False):
        # queue of Generators
        self.queue = initial
        # cuda active
        self.cuda = cuda
        self.test = "nothing special"
        self.notebook = notebook

        if self.queue is None:
            pass
        else:
            pass

    def __str__(self):
        #TODO implement
        back = ""
        return back

    def to_json(self):
        object = self
        return json.dumps(object)
