from ICImport.Countries import *
from ICImport.ICImport import *
from ICCluster.ICCluster import ICClusterSciKit as ICCluster

countries = {
    # 'brazil': ICImportBrazil,
    # 'paraguay': ICImportParaguay,
    # 'uruguay': ICImportUruguay,
    'argentina': ICImportArgentina
}

for country in countries:
    imports = countries[country](path="/home/ppcaunb/Downloads/trademap_"+country+".csv")
    imports.get_plot()
    plt.show()
    clusters = ICCluster(imports)
    print(clusters)
