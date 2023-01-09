
# =====================================================
## Vamos trabalhar com o modelo de Kolmorov Smirnorff para as nossas amostras
# =====================================================
# imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import kstest
from scipy.stats import ks_2samp
from matplotlib import pylab as plt
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openturns as ot
import openturns.viewer as viewer
import statsmodels.api as sm

# =====================================================
# Trabalhando com os dados da Dist Beta
#with open('dist_beta.txt') as f:
#    contents = f.read()
#    print(contents)

# https://www.youtube.com/watch?v=0NeAXFmrw4s&ab_channel=PaulMiskew
data_1 = open("dist_beta_2.txt","r")

dataString_1 = data_1.read()
#print(dataString_1)

dataList_1 = dataString_1.split("\n")
#print(dataList_1)

# Transformar todos os elementos em floats
#   0    1       2
# ["1","3.32","5.43"] 
for i in range(0, len(dataList_1),1):
    dataList_1[i] = dataList_1[i].replace(",","")
    dataList_1[i] = float(dataList_1[i])

#print(dataList_1)
# =====================================================
# Trabalhando com os dados da Dist Gamma 

data_2 = open("dist_gamma_2.txt","r")

dataString_2 = data_2.read()
#print(dataString_2)

dataList_2 = dataString_2.split("\n")
#print(dataList_2)

# Transformar todos os elementos em floats
#   0    1       2
# ["1","3.32","5.43"] 
for i in range(0, len(dataList_2),1):
    dataList_2[i] = dataList_2[i].replace(",","")
    dataList_2[i] = float(dataList_2[i])

#print(dataList_2)
# =====================================================
# Trabalhando com os dados da Dist Normal  

data_3 = open("dist_normal_2.txt","r")

dataString_3 = data_3.read()
#print(dataString_3)

dataList_3 = dataString_3.split("\n")
#print(dataList_3)

# Transformar todos os elementos em floats
#   0    1       2
# ["1","3.32","5.43"] 
for i in range(0, len(dataList_3),1):
    dataList_3[i] = dataList_3[i].replace(",","")
    dataList_3[i] = float(dataList_3[i])

#print(dataList_3)
# =====================================================
# Agrupar os dados 
dist_beta = dataList_1
dist_gamma = dataList_2
dist_normal = dataList_3
#f_a = np.random.f(dfnum = 5, dfden  = 10, size = 500)

dist_beta.sort()
dist_gamma.sort()
dist_normal.sort()
#f_a.sort()

# Visualizar as frequencias 
plt.figure(figsize = (10,3))
sns.histplot(dist_beta, bins = 20, kde = True, color = 'b')
sns.histplot(dist_gamma, bins = 20, kde = True, color = 'g')
sns.histplot(dist_normal, bins = 20, kde = True, color = 'r')
#sns.histplot(f_a, bins = 20, kde = True, color = 'orange')
plt.legend(["dist_beta", "dist_gamma", "dist_normal"])#, "f_a"])
plt.xlabel('Contador')
plt.ylabel('Frequencia')
plt.title('Analise das Distribuicoes')
plt.show()
# =====================================================
## Distribuicoes acumuladas amostrais

ot.Log.Show(ot.Log.NONE) ## reduzir freq de alarmes por motivo de logaritmos


x = dist_beta
#x = dist_gamma
#x = dist_normal
amostra = ot.Sample([[xi] for xi in x])

# %%
amostra_tamanho = amostra.getSize()
amostra_tamanho

# %%
# Distribuicao empirica

# %%
graph = ot.UserDefined(amostra).drawCDF()
graph.setLegends(["amostra"])
curve = ot.Curve([0, 1], [0, 1])
curve.setLegend("Uniforme")
graph.add(curve)
graph.setXTitle("X")
graph.setTitle("Distribuicao acumulada")
view = viewer.View(graph)
# =====================================================

# Dist entre a amostra e a dist computada com o teste ks.


# %%
def computar_ks(amostra, distribuicao):
    amostra = amostra.sort()
    n = amostra.getSize()
    D = 0.0
    D_ant = 0.0
    for i in range(n):
        F = distribuicao.computeCDF(amostra[i])
        Fminus = F - float(i) / n
        Fplus = float(i + 1) / n - F
        D = max(Fminus, Fplus, D)
        if D > D_ant:
            D_ant = D
    return D


# %%
dist = ot.Uniform(0.5, 1.6)
dist

# %%
computar_ks(amostra, dist)


# =====================================================
# Gerar as dist no teste ks para a distrubuicao

def gerar_param_ks(n_repeticoes, amostra_tamanho):

    dist = ot.Uniform(0.5, 1.6)#ot.Uniform(0, 1)
    D = ot.Sample(n_repeticoes, 1)
    for i in range(n_repeticoes):
        amostra = dist.getSample(amostra_tamanho)
        D[i, 0] = computar_ks(amostra, dist)
    return D

# =====================================================
# Gerar uma condensacao dos valores com base na amostragem (calibragem p os valores)

n_repeticoes = 1000  
amostraD = gerar_param_ks(n_repeticoes, amostra_tamanho)



# Computar a CDF (curva de distribuição da probabilidade) pelo teste KS.


def pKolmogorovPy(x):
    y = ot.DistFunc.pKolmogorov(amostra_tamanho, x[0])
    return [y]

pKolmogorov = ot.PythonFunction(1, 1, pKolmogorovPy)

# =====================================================
def dKolmogorov(x, amostra_tamanho): ## Curva PDF p o teste KS

    n = x.getSize()
    y = ot.Sample(n, 1)
    for i in range(n):
        y[i, 0] = pKolmogorov.gradient(x[i])[0, 0]
    return y

# =====================================================
# Variacao amostral para os valores max e mins.
def amostra_linear(xmin, xmax, npoints):

    step = (xmax - xmin) / (npoints - 1)
    rg = ot.RegularGrid(xmin, step, npoints)
    vertices = rg.getVertices()
    return vertices


# %%
n = 1000  # Numero de pontos plotados 
s = amostra_linear(0.001, 0.999, n)
y = dKolmogorov(s, amostra_tamanho)

# %%
curve = ot.Curve(s, y)
curve.setLegend("Distribuicao Exata")
graph = ot.HistogramFactory().build(amostraD).drawPDF()
graph.setLegends(["Distribuicao empirica"])
graph.add(curve)
graph.setTitle("Kolmogorov-Smirnov distribuicao")
graph.setXTitle("KS")
view = viewer.View(graph)



# =====================================================
# Parametros conhecidos versus estimados

def gerar_KS_amostras_est(n_repeticoes, amostra_tamanho):

    dist_fatorial = ot.UniformFactory()
    refdist = ot.Uniform(0, 1)
    D = ot.Sample(n_repeticoes, 1)
    for i in range(n_repeticoes):
        amostra = refdist.getSample(amostra_tamanho)
        trialdist = dist_fatorial.build(amostra)
        D[i, 0] = computar_ks(amostra, trialdist)
    return D

# =====================================================
# %%
# Gerar a amostra com as distancias pelo teste KS

amostraDP = gerar_KS_amostras_est(n_repeticoes, amostra_tamanho)

# %%
graph = ot.KernelSmoothing().build(amostraD).drawPDF()
graph.setLegends(["Parametros Conhecidos"])
graphP = ot.KernelSmoothing().build(amostraDP).drawPDF()
graphP.setLegends(["Parametros Estimados"])
graphP.setColors(["blue"])
graph.add(graphP)
graph.setTitle("Kolmogorov-Smirnov distribuicao")
graph.setXTitle("KS Estatisticas")
view = viewer.View(graph)
plt.show()

# =====================================================
## Testes entre as amostras ### Valores numericos

# Teste KS Beta
print("Teste KS para distribuicao beta:")
print(kstest(dist_beta, 'beta', [7, 10]))
#print(kstest(dist_beta, 'beta', args=(0,1)))
# =====================================================
# Teste KS Gamma
print("Teste KS para distribuicao gamma:")
print(kstest(dist_gamma, 'gamma', [7, 10]))
# =====================================================
# Teste KS Normal
print("Teste KS para distribuicao normal:")
print(kstest(dist_normal, 'norm'))
# =====================================================
# Teste para 2 amostras 

# Teste KS para dist Beta versus Gamma
print("Teste KS para dist Beta versus Gamma:")
print(ks_2samp(dist_beta, dist_gamma))
# =====================================================
# Teste KS para dist Beta versus Normal
print("Teste KS para dist Beta versus Normal:")
print(ks_2samp(dist_beta, dist_normal))
# =====================================================
# Teste KS para dist Normal versus Gamma
print("Teste KS para dist Normal versus Gamma:")
print(ks_2samp(dist_normal, dist_gamma))
# =====================================================