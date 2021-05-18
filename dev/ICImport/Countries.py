from ICImport.Codes import ICImportNCM
from ICImport.Sources import ICImportTradeMap


class ICImportBrazil(ICImportNCM, ICImportTradeMap):
    """
    Class for IC imports of Brazil
    """

    def __init__(self, df=None, path=None):
        super().__init__(df, path)
        self._import_source = 'Brazil'
