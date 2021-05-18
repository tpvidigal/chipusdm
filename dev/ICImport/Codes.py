from abc import ABC

from ICImport.ICImport import ICImport


class ICImportNCM(ICImport, ABC):
    """
    Abstract class for IC imports using NCM code
    """

    # NCM codes (since 2016)
    ncm_code = {
        '85423110': 'Processors and controllers, whether or not combined with memories, converters, logic circuits, amplifiers, clock and timing circuits, or other circuits - Unmounted',
        '85423120': 'Processors and controllers, whether or not combined with memories, converters, logic circuits, amplifiers, clock and timing circuits, or other circuits - Mounted, proper for surface mounting (SMD - Surface Mounted Device)',
        '85423190': 'Processors and controllers, whether or not combined with memories, converters, logic circuits, amplifiers, clock and timing circuits, or other circuits - Other',
        '85423210': 'Memories - Unmounted',
        '85423221': 'Memories - Mounted, proper for surface mounting (SMD - Surface Mounted Device) - Of types Static RAM (SRAM) with access time less or equal 25 ns, EPROM, EEPROM, PROM, ROM and FLASH',
        '85423229': 'Memories - Mounted, proper for surface mounting (SMD - Surface Mounted Device) - Other',
        '85423291': 'Memories - Other - Of types Static RAM (SRAM) with access time less or equal 25 ns, EPROM, EEPROM, PROM, ROM and FLASH',
        '85423299': 'Memories - Other - Other',
        '85423311': 'Amplificadores - Hybrid - With layer height less or equal to 1 micrometer (micron) with operation frequency greater or equal to 800MHZ',
        '85423319': 'Amplificadores - Hybrid - Other',
        '85423320': 'Amplificadores - Other, Unmounted',
        '85423390': 'Amplificadores - Other',
        '85423911': 'Other - Hybrid - With layer height less or equal to 1 micrometer (micron) with operation frequency greater or equal to 800MHZ',
        '85423919': 'Other - Hybrid - Other',
        '85423920': 'Other - Other, Unmounted',
        '85423931': 'Other - Other, Mounted, proper for surface mounting (SMD - Surface Mounted Device) - Circuits of type chipset',
        '85423939': 'Other - Other, Mounted, proper for surface mounting (SMD - Surface Mounted Device) - Other',
        '85423991': 'Other - Other - Circuits of type chipset',
        '85423999': 'Other - Other'
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

class ICImportCN(ICImport, ABC):
    """
    Abstract class for IC imports using Combined Nomenclature code
    """

    # CN codes (since 2017)
    cn_code = {
        '85423111': 'Processors and controllers, whether or  not  combined with  memories, converters, logic circuits, amplifiers, clock and timing circuits, or other circuits - note 9(b)(3 and 4) - Multi-component integrated circuits (MCOs)',
        '85423119': 'Processors and controllers, whether or  not  combined with  memories, converters, logic circuits, amplifiers, clock and timing circuits, or other circuits - note 9(b)(3 and 4) - Other',
        '85423190': 'Processors and controllers, whether or  not  combined with  memories, converters, logic circuits, amplifiers, clock and timing circuits, or other circuits - Other',
        '85423211': 'Memories - 9(b)(3 and 4) - Multi-component integrated circuits (MCOs)',
        '85423219': 'Memories - 9(b)(3 and 4) - Other',
        '85423231': 'Memories - Other - Dynamic random-access memories (D-RAMs) - With a storage capacity not exceeding 512 Mbits',
        '85423239': 'Memories - Other - Dynamic random-access memories (D-RAMs) - With a storage capacity exceeding 512 Mbits',
        '85423245': 'Memories - Other - Static random-access memories  (S-RAMs), including cache random-access memories (cache-RAMs)',
        '85423255': 'Memories - Other - UV erasable, programmable, read only memories (EPROMs)',
        '85423261': 'Memories - Other - Electrically erasable, programmable, read only memories (E²PROMs), including flash E²PROMs - Flash E²PROMs - With a storage capacity not exceeding 512 Mbits',
        '85423269': 'Memories - Other - Electrically erasable, programmable, read only memories (E²PROMs), including flash E²PROMs - Flash E²PROMs - With a storage capacity exceeding 512 Mbits',
        '85423275': 'Memories - Other - Electrically erasable, programmable, read only memories (E²PROMs), including flash E²PROMs - Other',
        '85423290': 'Memories - Other - Other memories',
        '85423310': 'Amplifiers - Multi-component integrated circuits (MCOs)',
        '85423390': 'Amplifiers - Other',
        '85423911': 'Other - 9(b)(3 and 4) - Multi-component integrated circuits (MCOs)',
        '85423919': 'Other - 9(b)(3 and 4) - Other',
        '85423990': 'Other - Other'
    }

    def __init__(self, df=None, path=None):
        super().__init__(df, path)
        self._code = 'CN'

    def get_code_dict(self):
        return ICImportCN.cn_code

    @staticmethod
    def code_description(code):
        """
        Get description of IC with given CN code
        :param code: CN code
        :return: Description of the IC
        """
        if code in ICImportCN.cn_code:
            return ICImportCN.cn_code[code]
        else:
            return "[DEPRECATED OR INVALID CODE]"
