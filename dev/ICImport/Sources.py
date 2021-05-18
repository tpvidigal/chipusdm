from abc import ABC

import pandas as pd

from ICImport.ICImport import ICImport


class ICImportReceitaFederal(ICImport, ABC):
    """
    Abstract class for IC imports with data from 'Receita Federal' of Brazil
    """

    def __init__(self, df=None, path=None):
        super().__init__(df, path)
        self._source = 'Receita Federal'

    def _get_data_from_file(self, path):
        return pd.read_csv(path, sep=';', engine='c', dtype={'CO_NCM': str, 'CO_ANO': int})

    def _prepare_data(self):

        # Select only integrated circuits from 2016 to now
        df = self._df_orig.loc[lambda row: row['CO_NCM'].isin(self._code_dict)]
        df = df[lambda row: row['CO_ANO'] >= 2016]

        # Sum total import volume per month
        df = df.groupby(['CO_NCM', 'CO_ANO', 'CO_MES']).sum().reset_index()

        # Add required columns
        df['DATE'] = df.apply(lambda row: "{:04d}".format(row['CO_ANO']) + '-' +
                                          "{:02d}".format(row['CO_MES']), axis=1)

        # Remove non-required columns
        df = df.loc[:, ['CO_NCM', 'KG_LIQUIDO', 'DATE']]
        df.rename(columns={'CO_NCM': 'CODE', 'KG_LIQUIDO': 'VOLUME'}, inplace=True)

        # Creating entries with 0 import volume for missing months for each chip
        df = df.set_index(['DATE', 'CODE'])['VOLUME'].unstack().unstack().reset_index()
        df = df.rename(columns={0: 'VOLUME'})
        df.fillna(0, inplace=True)

        # Transposing dates to transform importations into features of each chip
        df = df.pivot(index='DATE', columns='CODE', values='VOLUME')
        df.columns.name = None

        return df


class ICImportTradeMap(ICImport, ABC):
    """
    Abstract class for IC imports with data from TradeMap.org
    """

    def __init__(self, df=None, path=None):
        super().__init__(df, path)
        self._source = 'TradeMap.org'

    def _get_data_from_file(self, path):
        return pd.read_csv(path, sep=';', engine='c', dtype={'CODE': str})

    def _prepare_data(self):
        df = self._df_orig.loc[lambda row: row['CODE'].isin(self._code_dict)]
        df = df.set_index('CODE')
        df = df.T
        df.columns.name = None
        return df
