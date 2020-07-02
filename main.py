#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:52:21 2019

@author: iagorosa
"""


import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns

from sklearn.decomposition import PCA
#from sklearn.metrics.pairwise import pairwise_distances_argmin

from sklearn import cluster, metrics
from sklearn.preprocessing import StandardScaler

from itertools import product

from graficosNuvemPontos import *
from exploracaoDados import *
from stats import *

from caracterizacao import *

# In[12]:

# Leitura do arquivo gerado em preparacao-dados.py. 
T = pd.read_csv('arq_arrumado.csv')
T = T.iloc[:, 1:] # Remocao de indice 

# separacao dos atributos por tipos: numericos, cateoricos, binarios e identificadores
num_atributos = list(T.columns[5:14]) + list(T.columns[-6:-1])
cat_atributos = list(T.columns[1:3]) + ['bank-size', 'bank-size-index', 'specialisation_cat', 'country_code_cat']
bin_atributos= list(T.columns[17:29])
id_atributos = list(T.columns[0]) + list(T.columns[3])

# In[]

#DEFINICOES
group_size = ['Small Bank', 'Mid-size Bank', 'Large Bank', 'Super Large Bank']
color = ['red', 'green', 'blue', 'yellow', 'purple', 'pink']
anos = [2007, 2011]
forms = ['^', 's']
reg_cat = ['center', 'periphery']

bs_dict = pd.Series(T['bank-size'].values, index = T['bank-size-index'].values).to_dict()

cs = ['b', 'orange', 'g', 'r']
# Associacao de cores ao bank-size-index:
# {0: 'r', 1: 'orange', 2: 'g', 3: 'b'}
colors = dict(zip(T['bank-size-index'].unique(), [i for i in cs]))



# In[]

# Definicoes para de dataset para o pca e as clusterizacoes 

T_c = pd.concat([T[num_atributos[:-6]], T[bin_atributos[:3]+bin_atributos[-5:]], T[['year', 'bank-size', 'bank-size-index', 'index_number']]], axis=1).dropna()


Tcinbinatributes = list(set(bin_atributos).intersection(set(T_c.columns)))

# verifica as colunas seleciondas em T_c presentes em num_atributos para padroniza-los
Tcinnumatributes = list(set(num_atributos).intersection(set(T_c.columns)))

T_c = T_c[T_c['derivates'] >= 0]

T_c.loc[:, Tcinnumatributes] = np.log(T_c.loc[:, Tcinnumatributes] + 1)

scaler = StandardScaler().fit(T_c.loc[:, Tcinnumatributes])
T_c.loc[:, Tcinnumatributes] = scaler.transform(T_c.loc[:, Tcinnumatributes])

# In[]

def kmeans():
        
    # clusterizacao
    columnskmeans = Tcinnumatributes+Tcinbinatributes
    
    kmeans = cluster.KMeans(n_clusters=4)
    y_kmeans = kmeans.fit_predict(T_c.loc[:, columnskmeans])
    sil_kmeans = metrics.silhouette_score(T_c.loc[:, columnskmeans], y_kmeans, metric = 'euclidean')
    print('K-Means:')
    print('Rand Score:', metrics.adjusted_rand_score(T_c['bank-size-index'], y_kmeans))
    print('HCM:', metrics.homogeneity_completeness_v_measure(T_c['bank-size-index'], y_kmeans))
    print('Silhueta:', sil_kmeans)


    mbkmeans = cluster.MiniBatchKMeans(n_clusters=4, batch_size=20)   
    y = mbkmeans.fit_predict(T_c[columnskmeans])
    sil_mbkm = metrics.silhouette_score(T_c.loc[:, columnskmeans], y)
    
    print(sil_mbkm)

    return y_kmeans if sil_kmeans > sil_mbkm else y

#kmeans()

# In[]

def cluster_hier(x, n_clusters = 4, pnt = True):

    m_sil = []
    y_predito = []
    
    linkage = ['single', 'complete', 'average']
    affinity = ['l1', 'l2', 'cosine']
    
    for k in range(4, n_clusters+1):
        for i, j in product(linkage, affinity):
            try:
                modelo = cluster.AgglomerativeClustering(n_clusters=k, linkage=i, affinity=j)

                y_predito.append(modelo.fit_predict(x))
                
                m_sil.append( (metrics.silhouette_score(x, y_predito[-1], metric = j), i, j) )

                print(i, j, k, m_sil[-1])
            except:
                pass
            
    
    id = m_sil.index(max(m_sil))
    
    

    if pnt:
        print ('\nMelhor usando AgglomerativeClustering com %s: '%m_sil[id][1])
        print ('silhouette_score: ', m_sil[id])
        
        media = np.array(m_sil)
        media = media[:, 0].astype(float).mean()
        
    return m_sil[id], y_predito[id], m_sil


# In[]

def dbscan(x, eps = 0.5, min_sample = 5):
    modelo = cluster.DBSCAN(eps=eps, min_samples=min_sample)

    # treina o modelo e gera os valores de resposta para os grupos
    y_predito = modelo.fit_predict(x)
    
    return metrics.silhouette_score(x, y_predito, metric = 'euclidean'), y_predito
    

def pardbscan(x, mbs = 10, prm_eps=[0.1, 3, 0.1], prm_min_sample = [2, 1010, 10]):
        best_dbscan = 0
        best_y = 0
        best_eps= 0
        best_min_sample = 0
        for i in np.arange(prm_eps[0], prm_eps[1], prm_eps[2]):
            if i != 0.1: i -= 0.1
            for j in np.arange(prm_min_sample[0], prm_min_sample[1], prm_min_sample[2]):
                                
                if j != 1: j -= 2

                try:
                    aux = dbscan(x, eps=i, min_sample=j)
                    if aux[0] > 0.1 and len(np.unique(aux[1])) > 2:
                        print(round(i, 2), j, round(aux[0], 4), len(np.unique(aux[1])))
                except: 
                    pass
                else:
                    if aux[0] > best_dbscan and len(np.unique(aux[1])) > 2:
                        best_dbscan = aux[0]  
                        best_y = aux[1]
                        best_eps = i
                        best_min_sample = j
                        
        print(best_eps, best_min_sample)

        best_y = best_y + (1 if -1 in best_y else 0)
        print('Quantidade estimada de cluster:', len(np.unique(best_y)))
        
        print(best_dbscan)
                        
        best_y = best_y + (1 if -1 in best_y else 0)        

        return best_dbscan, best_y
        

# In[]

def kmeansteste(x, n_cluster = 4, N=10):
        
    # clusterizacao    
    bestkmeans = 0
    bestsilkmeans = 0
    
    valores_sil = []
    
#    bestmbkmeans = 0
#    bestsilmbkmeans = 0
    
    for i in range(N):
        kmeans = cluster.KMeans(n_clusters=n_cluster)
        y_kmeans = kmeans.fit_predict(x)
        sil_kmeans = metrics.silhouette_score(x, y_kmeans, metric = 'euclidean')
        
#        mbkmeans = cluster.MiniBatchKMeans(n_clusters=4, batch_size=20)   
#        y = mbkmeans.fit_predict(x)
#        sil_mbkm = metrics.silhouette_score(x, y)
        
        bestkmeans, bestsilkmeans = (y_kmeans, sil_kmeans) if sil_kmeans > bestsilkmeans else (bestkmeans, bestsilkmeans)
#        bestmbkmeans, bestsilmbkmeans  = y, sil_mbkm if sil_mbkm > bestsilmbkmeans else bestmbkmeans, bestsilmbkmeans

#    print('K-Means:')
#    print('Rand Score:', metrics.adjusted_rand_score(T_c['bank-size-index'], y_kmeans))
#    print('HCM:', metrics.homogeneity_completeness_v_measure(T_c['bank-size-index'], y_kmeans))
#    print('Silhueta:', sil_kmeans)
        
        valores_sil.append(sil_kmeans)
    
    valores_sil = np.array(valores_sil)
    print('Media:', valores_sil.mean())
    print('Devpad:', valores_sil.std())
    print('Melhor:', bestsilkmeans)    
#    return y_kmeans if sil_kmeans > sil_mbkm else y
    return y_kmeans

# In[]

def pca(y_predito = None, k = 4):
    
    colunaspca = Tcinnumatributes +Tcinbinatributes
    
#    T_c[Tcinnumatributes].plot()
#    T_c[Tcinnumatributes] = np.log(T_c[Tcinnumatributes])
    
    t = ['Pequeno', 'Médio', 'Grande', 'Super Grande']
    tam = dict(zip(range(5), t))    
    
#    print(T_c.loc[:, colunaspca].isna().sum())
    
    # PCA para visualizacao
    pca = PCA(n_components=4)
    X_pca = pca.fit(T_c.loc[:, colunaspca]).transform(T_c.loc[:, colunaspca])
    ## PCA process results
    results = pca.components_
    covm = pca.explained_variance_ratio_
    
    if y_predito is None:
#        print( '\n\n\n\n\n\entrou\n\n\n\n\n\n')
#        y_predito = kmeansteste(X_pca, n_cluster=3)
        y_predito = cluster_hier(X_pca)[1]
#        pardbscan(X_pca)
#        sil, y_predito = dbscan(X_pca, eps=0.7, min_sample=110)
#        y_predito = y_predito + (1 if -1 in y_predito else 0)
#        print('\n\n\n\n'+str(sil)+'\n\n\n')
#        print(y_predito)
#        y_predito = y_db
    
#    minimo0 = X_pca[:, 0].min() 
#    minimo1 = X_pca[:, 1].min()  
#    eps = 0.0001
    
#    X_pca = np.concatenate((X_pca, T_c.year.values.reshape(len(T_c.year), 1)), axis=1)
    
    print("Explained variance ratio:", covm)
#    print(X_pca)
    
    vn1 = [(i, j) for i, j in zip(T_c[colunaspca].columns, results[0])]
    vn2 = [(i, j) for i, j in zip(T_c[colunaspca].columns, results[1])]
    # vn3 = [(i, j) for i, j in zip(T.iloc[:,:-2].columns, results[2])]
    
    vn1.sort(key=lambda x: -abs(x[1]))
    vn2.sort(key=lambda x: -abs(x[1]))
    # vn3.sort(key=lambda x: -abs(x[1]))
    
    print(vn1)
    print("\n", vn2)
    
#    pl.figure(figsize=(10,6))
    
    colors_aux = ['r', 'y', 'g', 'b', 'pink', 'c', 'orange', 'silver', 'indigo', 'maroon']
#    pl.figure(figsize=(12,6))
    pl.figure()
    #X_pca_scatter = pd.concat([pd.DataFrame(X_pca), banksize, labelbs], axis=1)
    
    #X_pca_scatter = pd.concat([pd.DataFrame(X_pca), pd.Series(y_kmeans, name='cluster')], axis=1)
    
    X_pca_scatter = pd.concat([pd.DataFrame(X_pca), T_c.iloc[:, -3:].reset_index(drop=True), T_c.assets.reset_index(drop=True), T_c.year.reset_index(drop=True), pd.Series(y_predito, name='cluster')], axis=1)
#    
#    pl.subplot(1, 2, 1)
#    for i, df in X_pca_scatter.groupby('cluster'):
#        pl.scatter(df.iloc[:,0], df.iloc[:,1], marker='o', c=colors[int(i)], label='cluster '+str(i), s=25, edgecolor='k')
    
    colors_cluster = dict(zip(X_pca_scatter.groupby('cluster').mean()['assets'].sort_values().index, [i for i in colors_aux[:max(y_predito)+1]]))
    
    print(colors_aux[:max(y_predito)+1])
    print(colors_cluster)
    
#    pl.subplot(1, 2, 1)
#    pl.figure()
    for i, df in X_pca_scatter.groupby('cluster'):
#        pl.scatter(df.iloc[:,0], df.iloc[:,1], marker='o', c=colors_aux[int(i)], label='cluster '+str(i), s=25, edgecolor='k')
        pl.scatter(df.iloc[:,0], df.iloc[:,1], marker='o', c=colors_cluster[int(i)], label=str(i), s=25, edgecolor='k')
        
    pl.title('Resultado ACP + ' + r'DBSCAN')
    pl.legend(title='Cluster')
#    pl.xscale('log')
#    pl.yscale('log')
    #pl.title('PCA pós processamento')
    pl.ylabel(r'$ACP \ Z_2$', )
    pl.xlabel(r'$ACP \ Z_1$', )
    # fig.tight_layout()
    pl.savefig('imgs/Caracterizacao/pca.eps', dpi=600)
    #pl.show()
    

    return X_pca_scatter

# In[]

def separacaoPorAtivo(X_pca_scatter):
    
    cs = ['r', 'y', 'g', 'b']
    # Associacao de cores ao bank-size-index:
    # {0: 'r', 1: 'orange', 2: 'g', 3: 'b'}
    colors = dict(zip(sorted(T['bank-size-index'].unique()), [i for i in cs]))
    
    t = ['Pequeno', 'Médio', 'Grande', 'Super Grande']
    tam = dict(zip(range(5), t))  

    X_aval = pd.DataFrame()
    
    for y in sorted(T.year.unique()):
        X_aval = pd.concat([X_aval, pd.merge(T[T.year == y], X_pca_scatter[X_pca_scatter.year == y][['index_number', 'cluster']], how='inner', on='index_number')], axis=0)
    
    colors_cluster = dict(zip(X_aval.groupby('cluster').mean()['assets'].sort_values().index, [i for i in cs]))
    
    minval = X_aval.groupby('cluster').agg('min')['assets']
    maxval = X_aval.groupby('cluster').agg('max')['assets']
    
#    bins = [0, 1e+7, 5e+7, 2.5e+8, np.finfo(T['assets'].dtype).max ]
    
    newbins = pd.concat([minval.rename('min'), maxval.rename('max')], axis=1)
    
    # vals = sorted(newbins.get_values().reshape(len(newbins.columns)*len(newbins)))
    
    if all(newbins['min'] < newbins['max']):
        newclassbanks = [0] + sorted(newbins['min']) +[np.finfo(X_aval['assets'].dtype).max]
        
#        print('sim')
        newclassbanks = ["{:.2e}".format(num) for num in newclassbanks]
    
#    pl.figure(figsize=(10,6))
#    pl.figure()
    
    pl.subplot(2, 1 , 1)
    
    for x, df in X_aval.groupby('bank-size-index'):
        sns.kdeplot(np.log(df['assets']), label = tam[x], shade=1,lw=2, color=colors[x])
    pl.legend(title='Tamanho do banco')
    pl.title('Estimativa de Densidade de Probabilidade do Ativo -- ACP')
    '''
    
    pl.subplot(2, 1, 2)
    for x, df in X_aval.groupby('cluster'):
        sns.kdeplot(np.log(df['assets']), label = str(x), shade=1,lw=2, color=colors_cluster[x])
       '''
#    pl.title('Estimativa de Densidade de Probabilidade do Ativo \n Hierárquico Aglomerativo')
#    pl.legend(title='Cluster')
    pl.tight_layout()
    pl.savefig('imgs/Caracterizacao/comparacao_bank_size.png', dpi=400)
    pl.show()
    
    return X_aval
        
        
# In[]


#graficoNuvemPontosPorAno(T, show = True)
#graficoNuvemPontosPorAnoRegional(T, show=True)
#graficoNuvemPontosGeral(X_aval, anos=[2007,2011], show=True)
#movimentoAnualBancosGeral(X_aval, show=True, coluna = 'cluster')
#movimentoAnualBancosRegional(X_aval, show = True, coluna='cluster')
#densidadeDeProbabilidadeGeral(X_aval, show = True, coluna = 'cluster')
#densidadeDeProbabilidadeRegional(T, show = True)
#bxp = boxplotGeralPorAno(X_aval)
#boxplotRegional()
#centroides = distancias()
#graficoDistanciasGeral(X_aval, coluna='cluster')
#graficoDistanciasRegional(X_aval, coluna='cluster')
  
#kld = heat_map_Geral(T, kld = False, show=True)
#graficodistanciakld(X_aval, kld, colClusterClusteruna='cluster', show = True)
#heat_map_Regional()
pl.rcParams.update(pl.rcParamsDefault)

# Criacao dos cluster e agrupamento com o dataset original.
# Clusterizacao padrao do pca eh a kmClustereans. Caso queira outra, passsa-la como parametro.
X_pca_scatter = pca()
X_aval = separacaoPorAtivo(X_pca_scatter)

#y_k = kmeansteste(T_c[Tcinbinatributes+Tcinnumatributes], n_cluster=3, N=10)
#sil, y_k, m_sil = cluster_hier(T_c[Tcinnumatributes+Tcinbinatributes], n_clusters=3)
#pca(y_k)


#X_aval = pd.merge(T_c, X_pca_scatter[['index_number', 'cluster']], how='inner', left_on='index_number', right_on='index_number')

#boxplotGeralPorAno(X_aval, show = True)
#boxplotRegional(X_aval, coluna='cluster', show=True)

#T_c.iloc[:, :-4].to_csv('data_bankscope.csv')
#T_c['bank-size-index'].to_csv('target_bankscope.csv')
# In[]

##Conferindo cluster por assets:


#X_aval = pd.concat(T_c, X_pca_scatter[['index_number', 'cluster']], axis=1)


#minval = X_aval[(X_aval.cluster==0) & (X_aval.year == 2007)].groupby('bank-size').agg('min')['assets']
#maxval = X_aval[(X_aval.cluster==0) & (X_aval.year == 2007)].groupby('bank-size').agg('max')['assets']

T[T.year == 2006]


#X_aval = separacaoPorAtivo(X_pca_scatter)

# In[]

#def boxplotTamanhoAno(): 
def testesEstatisticos():
    
#    dff = []
#    anos = []
#    rests = []
#    for x, df in T.groupby('bank-size'):
#        fig = pl.figure()
#        classnames, indices = np.unique(group_size, return_inverse=True)  
#        for v in ['AL']:
#            for y,df1 in df.groupby('year'):
#                anos.append(y)
#                dff.append(df1[v])
#        ax = fig.add_subplot(111)
#        ax.boxplot(dff, labels=anos)
#        ax.set_title(x)
#        if x == 'Small Bank':
#            rests.append(dff)
#        dff = []
#        anos = []
        
    test = []
    anos = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    for a in anos:
        test.append(np.array(T[(T['bank-size'] == 'Small Bank') & (T['year'] == a) & (T['periphery'] == 1)]['AL']))
    
    
#    test = []
#    anos = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
#    for a in anos:
#        test.append(np.array(T[(T['bank-size'] == 'Small Bank') & (T['year'] == a)]['AL']))  
    



def setas():
    pl.figure(2)
    j=0
    k=0
    setas = np.zeros((4, 4))
    for y, df in T.groupby('year'):
        if (y in anos):
    #        classnames, indices = np.unique(df['bank-size'].get_values(), return_inverse=True)
            classnames, indices = np.unique(group_size, return_inverse=True)
            for i in sorted(np.unique(indices)):
                aux = df[(df['year'] == y) & (df['bank-size'] == classnames[i])]
#                yerr_l = abs(min(aux['AL']) - aux['AL'].mean())
#                yerr_u = abs(max(aux['AL']) - aux['AL'].mean())
    #            print(aux['AL'].mean(), yerr_u)
    #            print([yerr_l, yerr_u])
                setas[i][k]   = aux['AL'].mean()
                setas[i][k+1] = aux['PL'].mean()
                pl.scatter(aux['AL'].mean(), aux['PL'].mean(), marker=forms[j], c=color[i], label = classnames[i], s = 50)#, yerr=0.1, xerr=[yerr_l, yerr_u])
    #                        yerr=[yerr_l, yerr_u])#, 
    #                        marker=forms[j], c=color[i], label = classnames[i])
            pl.xlabel('AL');pl.ylabel('PL')
            pl.axhline(0,ls='-.');pl.axvline(0,ls='-.')
    #        pl.show()
            j += 1
            k += 2
    
    for i in range(4):
        pl.arrow(setas[i][0], setas[i][1], (setas[i][2]-setas[i][0]), (setas[i][3]-setas[i][1]), shape='full', lw=0, length_includes_head=True, head_width=.02, fc=color[i], ec=color[i])
    
    #pl.arrow(0.3, -0.55, 0.45-0.3, -0.44+0.55)#, head_width=0.05, head_length=0.1, fc='k', ec='k')
    
    pl.legend(loc='best')
    pl.title(anos)
    pl.show()
    
