from ICImport.Countries import *
from ICImport.ICImport import *

#from ICCluster.ICClusterMadsonDias import ICClusterMadsonDias as ICCluster
from ICCluster.ICClusterSciKit import ICClusterSciKit as ICCluster

countries = {
    'brazil': (ICImportBrazil, "/home/ppcaunb/Downloads/receita_brasil.csv"),
    'paraguay': (ICImportParaguay, "/home/ppcaunb/Downloads/trademap_paraguay.csv"),
    'uruguay': (ICImportUruguay, "/home/ppcaunb/Downloads/trademap_uruguay.csv"),
    'argentina': (ICImportArgentina, "/home/ppcaunb/Downloads/trademap_argentina.csv"),
    'germany': (ICImportGermany, "/home/ppcaunb/Downloads/trademap_germany.csv")
}

for country in countries:
    imports = countries[country][0](path=countries[country][1])
    imports.get_plot()
    plt.show()
    clusters = ICCluster(imports)
    print(clusters)
