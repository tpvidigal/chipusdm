import matplotlib.pyplot as plt
import pandas as pd


class ICImport:
    """
    Class that encapsulates the IC importations
    """

    # NCM dictionary of codes (since 2016)
    ncm_dict = {
        '85423110': 'Processadores e controladores, mesmo combinados com memórias, conversores, circuitos lógicos, amplificadores, circuitos temporizadores e de sincronização, ou outros circuitos - Não Montados',
        '85423120': 'Processadores e controladores, mesmo combinados com memórias, conversores, circuitos lógicos, amplificadores, circuitos temporizadores e de sincronização, ou outros circuitos - Montados, próprios para montagem em superfície (SMD - Surface Mounted Device)',
        '85423190': 'Processadores e controladores, mesmo combinados com memórias, conversores, circuitos lógicos, amplificadores, circuitos temporizadores e de sincronização, ou outros circuitos - Outros',
        '85423210': 'Memórias - Não Montadas',
        '85423221': 'Memórias - Montadas, próprios para montagem em superfície (SMD - Surface Mounted Device) - Dos tipos RAM estáticas (SRAM) com tempo de acesso inferior ou igual a 25 ns, EPROM,EEPROM, PROM, ROM e FLASH',
        '85423229': 'Memórias - Montadas, próprios para montagem em superfície (SMD - Surface Mounted Device) - Outras',
        '85423291': 'Memórias - Outras - Dos tipos RAM estáticas (SRAM) com tempo de acesso inferior ou igual a 25 ns, EPROM, EEPROM, PROM, ROM e FLASH',
        '85423299': 'Memórias - Outras - Outras',
        '85423311': 'Amplificadores - Híbridos - De espessura de camada inferior ou igual a 1 micrômetro (mícron) com frequência de operação superior ou igual a 800 MHz',
        '85423319': 'Amplificadores - Híbridos - Outros',
        '85423320': 'Amplificadores - Outros, não montados',
        '85423390': 'Amplificadores - Outros',
        '85423911': 'Outros - Híbridos - De espessura de camada inferior ou igual a 1 micrômetro (mícron) com frequência de operação superior ou igual a 800 MHz',
        '85423919': 'Outros - Híbridos - Outros',
        '85423920': 'Outros - Outros, não montados',
        '85423931': 'Outros - Outros, montados, próprios para montagem em superfície (SMD - Surface Mounted Device) - Circuitos do tipo chipset',
        '85423939': 'Outros - Outros, montados, próprios para montagem em superfície (SMD - Surface Mounted Device) - Outros',
        '85423991': 'Outros - Outros - Circuitos do tipo chipset',
        '85423999': 'Outros - Outros'
    }

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
        df = self._df_orig.loc[lambda row: row['CO_NCM'].isin(ICImport.ncm_dict)]
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

    def get_num_ics(self):
        """
        Return number of ICs recorded
        :return: Integer with number of ICs
        """
        return len(self._df.columns)

    def get_ics(self):
        """
        Return ICs recorded
        :return: List of strings of IC codes
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

    @staticmethod
    def ncm_to_description(ncm):
        """
        Get description of IC with given NCM code
        :param ncm: NCM code
        :return: Description of the IC
        """

        if ncm in ICImport.ncm_dict:
            return ICImport.ncm_dict[ncm]
        else:
            return "[DEPRECATED OR INVALID CODE]"
