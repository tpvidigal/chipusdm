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


class ICImportUruguay(ICImportNCM, ICImportTradeMap):
    """
    Class for IC imports of Uruguay
    """

    def __init__(self, df=None, path=None):
        super().__init__(df, path)
        self._country = 'Uruguay'


class ICImportArgentina(ICImportNCM, ICImportTradeMap):
    """
    Class for IC imports of Argentina
    """

    def __init__(self, df=None, path=None):
        super().__init__(df, path)
        self._country = 'Argentina'
