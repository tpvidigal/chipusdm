from ICImport.Codes import ICImportNCM
from ICImport.Sources import ICImportTradeMap


class ICImportBrazil(ICImportNCM, ICImportTradeMap):
    """
    Class for IC imports of Brazil
    """

    def __init__(self, df=None, path=None):
        super().__init__(df, path)
        self._country = 'Brazil'


class ICImportParaguay(ICImportNCM, ICImportTradeMap):
    """
    Class for IC imports of Paraguay
    """

    def __init__(self, df=None, path=None):
        super().__init__(df, path)
        self._country = 'Paraguay'
