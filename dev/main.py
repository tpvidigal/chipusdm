from ICImport.Countries import *
from ICImport.ICImport import *
from functools import reduce
import numpy as np

#from ICCluster.ICClusterMadsonDias import ICClusterMadsonDias as ICCluster
from ICCluster.ICClusterSciKit import ICClusterSciKit as ICCluster

countries = {
    'brazil': (ICImportBrazil, "/home/ppcaunb/Downloads/receita_brasil.csv"),
    'paraguay': (ICImportParaguay, "/home/ppcaunb/Downloads/trademap_paraguay.csv"),
    'uruguay': (ICImportUruguay, "/home/ppcaunb/Downloads/trademap_uruguay.csv"),
    'argentina': (ICImportArgentina, "/home/ppcaunb/Downloads/trademap_argentina.csv"),
    'germany': (ICImportGermany, "/home/ppcaunb/Downloads/trademap_germany.csv"),
    'france': (ICImportFrance, "/home/ppcaunb/Downloads/trademap_france.csv"),
    'belgium': (ICImportBelgium, "/home/ppcaunb/Downloads/trademap_belgium.csv")
}

clusters = {}
for country in countries:
    imports = countries[country][0](path=countries[country][1])
    imports.get_plot()
    plt.show()
    clusters[country] = ICCluster(imports)
    # print(clusters, '\n')

countries_ncm = [c for c in countries if issubclass(countries[c][0], ICImportNCM)]
countries_cn = [c for c in countries if issubclass(countries[c][0], ICImportCN)]
clusters_ncm = [clusters.get(c) for c in countries_ncm]
clusters_cn = [clusters.get(c) for c in countries_cn]
mean_ncm = reduce(lambda x, y: x.add(y), [c.get_fuzzy_partition() for c in clusters_ncm]) / len(clusters_ncm)
mean_cn = reduce(lambda x, y: x.add(y), [c.get_fuzzy_partition() for c in clusters_cn]) / len(clusters_cn)
cvmean_ncm = np.mean([c.get_cross_validation().get_eval() for c in clusters_ncm])
cvmean_cn = np.mean([c.get_cross_validation().get_eval() for c in clusters_cn])
cvstd_ncm = np.std([c.get_cross_validation().get_eval() for c in clusters_ncm])
cvstd_cn = np.std([c.get_cross_validation().get_eval() for c in clusters_cn])
print("")
print("NCM mean Fuzzy Partition")
print("Validity:", str(cvmean_ncm), '+-', str(cvstd_ncm))
print(mean_ncm.to_string())
print("")
print("CN mean Fuzzy Partition")
print("Validity:", str(cvmean_cn), '+-', str(cvstd_cn))
print(mean_cn.to_string())
