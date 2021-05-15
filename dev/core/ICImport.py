import matplotlib.pyplot as plt
import pandas as pd


class ICImport:
    """
    Class that encapsulates the IC importations
    """

    def __init__(self, df=None, path=None):
        if df is None:
            if path is not None:
                df = pd.read_csv(path, sep=';', engine='c', dtype={'CO_NCM': str})
        self._df_orig = df
        self._df = self._prepare_data()
        self._df_norm = self._normalize_date()

    def _prepare_data(self):
        """
        Data preparation of original data
        :return: Dataframe with prepared data
        """
        if self._df_orig is None:
            return None

        # Select only integrated circuits from 2016 to now
        df = self._df_orig.loc[lambda row: row['CO_NCM'].str.startswith('8542')]
        df = df[lambda row: row['CO_ANO'] >= 2016]

        # Sum total import volume per month
        df = df.groupby(['CO_NCM', 'CO_ANO', 'CO_MES']).sum().reset_index()

        # Add required columns
        df['CO_SH6'] = df['CO_NCM'].str[0:6]
        df['DATE'] = df.apply(lambda row: "{:04d}".format(row['CO_ANO']) + '-' +
                                          "{:02d}".format(row['CO_MES']), axis=1)

        # Remove non-required columns
        df = df.loc[:, ['CO_NCM', 'QT_ESTAT', 'DATE']]
        df.rename(columns={'CO_NCM': 'CODE', 'QT_ESTAT': 'VOLUME'}, inplace=True)

        # Creating entries with 0 import volume for missing months for each chip
        df = df.set_index(['DATE', 'CODE'])['VOLUME'].unstack().unstack().reset_index()
        df = df.rename(columns={0: 'VOLUME'})
        df.fillna(0, inplace=True)

        # Transposing dates to transform importations into features of each chip
        df = df.pivot(index='DATE', columns='CODE', values='VOLUME')
        df.columns.name = None

        return df

    def _normalize_date(self):
        """
        Normalize prepared data
        :return: Dataframe with normalized prepared data
        """
        if self._df is None:
            return None

        # Normalizing import volume: Min-Max
        df = self._df.copy().reset_index().set_index('DATE')
        for ic in df:
            ic_min = df[ic].min()
            ic_max = df[ic].max()
            df[ic] = df[ic].apply(lambda x: (x - ic_min) / (ic_max - ic_min))

        return df

    def get_data(self):
        """
        Get data for clustering operation
        :return: Dataframe with data ready for clustering
        """
        return self._df_norm

    def get_num_records(self):
        """
        Return number of month records
        :return: Integer with number of records
        """
        return len(self._df)

    def get_ics(self):
        """
        Return number of ICs recorded
        :return: Integer with number of ICs
        """
        return self._df.columns

    def get_plot(self):
        """
        Plot imports data
        :return: Axes object of the plot
        """
        if self._df is not None:
            ax = self._df.plot(figsize=(8, 6))
            ax.set_title('IC Importations in Brazil')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume of imports')
            plt.legend(bbox_to_anchor=(1.02, 1.05), loc='upper left', borderaxespad=0.)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=0.9)
            return ax

    def get_plot_norm(self):
        """
        Plot imports data normalized
        :return: Axes object of the plot
        """
        if self._df is not None:
            ax = self._df_norm.plot(figsize=(8, 6))
            ax.set_title('IC Importations in Brazil')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume of imports (normalized)')
            plt.legend(bbox_to_anchor=(1.02, 1.05), loc='upper left', borderaxespad=0.)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=0.9)
            return ax
