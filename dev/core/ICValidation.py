import numpy as np


class ICValidation:

    def __init__(self, fpc=None):
        self._fpc = fpc
        self._fpc_std = 0

    def __str__(self):
        info = "IC Validation info:\n"
        info += "  '-> FPC: " + str(self._fpc)
        if self._fpc_std > 0:
            info += " +- " + str(self._fpc_std)
        return info

    def mean_validity(self, validations):
        fpc = []
        for result in validations:
            fpc += [result.get_fpc()]
        self._fpc = np.mean(fpc)
        self._fpc_std = np.std(fpc)

    def get_eval(self):
        return self.get_fpc()

    def set_fpc(self, fpc):
        self._fpc = fpc

    def get_fpc(self):
        return self._fpc
