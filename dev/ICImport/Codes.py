from abc import ABC

from ICImport.ICImport import ICImport


class ICImportNCM(ICImport, ABC):
    """
    Abstract class for IC imports using NCM code
    """

    # NCM codes (since 2016)
    ncm_code = {
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
        super().__init__(df, path)
        self._code = 'NCM'

    def get_code_dict(self):
        return ICImportNCM.ncm_code

    @staticmethod
    def code_description(code):
        """
        Get description of IC with given NCM code
        :param code: NCM code
        :return: Description of the IC
        """
        if code in ICImportNCM.ncm_code:
            return ICImportNCM.ncm_code[code]
        else:
            return "[DEPRECATED OR INVALID CODE]"
