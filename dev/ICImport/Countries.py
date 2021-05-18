from ICImport.Codes import *
from ICImport.Sources import *


class ICImportBrazil(ICImportNCM, ICImportReceitaFederal):
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


class ICImportGermany(ICImportCN, ICImportTradeMap):
    """
    Class for IC imports of Germany
    """

    def __init__(self, df=None, path=None):
        super().__init__(df, path)
        self._country = 'Germany'
