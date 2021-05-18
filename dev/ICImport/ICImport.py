import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class ICImport(ABC):
    """
    Abstract class that encapsulates the IC importations
    """

    def __init__(self, df=None, path=None):
        if df is None:
            if path is not None:
                df = self._get_data_from_file(path)

        self._import_source = '<?>'
        self._code_dict = self.get_code_dict()
        self._df_orig = df
        self._df = self._prepare_data()
        self._df_norm = self._normalize_data()

    def _normalize_data(self):
        if self._df is None:
            return None

        # Normalizing import volume: Min-Max
        df = self._df.copy()
        for ic in df:
            ic_min = df[ic].min()
            ic_max = df[ic].max()
            df[ic] = df[ic].apply(lambda x: (x - ic_min) / (ic_max - ic_min))

        return df

    @abstractmethod
    def _get_data_from_file(self, path):
        pass

    @abstractmethod
    def _prepare_data(self):
        """
        Data preparation of original data
        :return: Dataframe with prepared data
        """
        pass

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
            ax.set_title('IC Importations in ' + self._import_source)
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
            ax.set_title(self._import_source)
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume of imports (normalized)')
            plt.legend(bbox_to_anchor=(1.02, 1.05), loc='upper left', borderaxespad=0.)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=0.9)
            return ax

    @abstractmethod
    def get_code_dict(self):
        pass

    @staticmethod
    @abstractmethod
    def code_description(code):
        pass
