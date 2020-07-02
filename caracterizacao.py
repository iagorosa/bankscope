#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:49:28 2019

@author: iagorosa
"""

import pandas as pd
import numpy as np

import pylab as pl

import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#from scipy.stats import norm
#import matplotlib.mlab as mlab

# In[]

def renameNumericAtributes():
    abrev = {'employees': 'Quantidade de \nFuncionários', 'loans': 'Empréstimos', 'LAA': "Adiantamentos", 'derivates': 'Derivativos', 'assets': 'Ativo', 'DAST': 'Financ. de \nCurto Prazo', 'DFB': 'Depósitos\n de bancos', 'LTF': 'Financ. de\nLongo Prazo', 'TLAE': 'Passivo', 'bank-size': 'Bank Size'}
    
#    abrev_array = np.array(sorted(abrev.items(), key=lambda x: x[1]))
    
    return abrev

# In[]
    
def tamanho():
    t = ['Pequeno', 'Médio', 'Grande', 'Super Grande']
    return dict({'Small Bank': t[0], 'Mid-size Bank': t[1], 'Large Bank': t[2], 'Super Large Bank': t[3]}) 

def localizacao():
    return dict({'periphery': 'Periferia', 'others': 'Outros', 'center': 'Centro'})

# In[]

def dadosFaltantesPorIntanciaAno(X, show=False):
    ### TABELA DE QUANTIDADE DE NAN POR INSTÂNCIA/ANO
    aux = X.isna().sum(axis=1)
    aux = pd.concat([aux, X['year']], axis=1)
    aux = aux.groupby([0, 'year'])['year']
    nan_inst = aux.count().unstack().fillna(0).astype(int)
    
    nan_inst = nan_inst.rename_axis('qtd')
#    nan_inst.sort_index(ascending=False, inplace=True) #Comentar pra inverter
    
    total = nan_inst.sum(axis=1)
    ptg = total/sum(total)
    
    nan_inst = pd.concat([nan_inst, total.rename('total'), ptg.rename('%')], axis=1)
    
    nan_inst['% acum'] = nan_inst['%']
    
    nan_inst['% acum'] = nan_inst['%'].cumsum()
    
    nan_inst['%'] *= 100
    nan_inst['% acum'] *= 100
         
    fig, ax1 = pl.subplots()
    pl.grid(axis='y', zorder=0)
    
#    ax1.set_title('Dados faltantes', fontsize=18, fontweight='bold')
#    ax1.set_xlabel('Quantidade', fontsize=14, fontweight='bold')
#    ax1.set_ylabel('Frequência relativa (%)', color='blue', fontsize=14, fontweight='bold')
#    ax1.tick_params(axis='x', labelsize=12)
#    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_title('Frequência relativa e acumulada de dados faltantes')
    ax1.set_xlabel('Quantidade')
    ax1.set_ylabel('Frequência relativa (%)', color='blue')
    ax1.bar(nan_inst.index, nan_inst['%'], color='blue', zorder=3)
    ax1.tick_params(axis='y', labelcolor='blue') 
    
    ax2 = ax1.twinx()  
    
    ax2.plot(nan_inst['% acum'],  'rs-')
#    ax2.tick_params(axis='y', labelcolor='red', labelsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel('Frequência acumulada (%)', color='red')
    pl.setp(ax2, yticks=np.arange(0, 101, 10))
    
    fig.tight_layout()  
    pl.xticks(np.arange(0, 11))
#    pl.gca().invert_xaxis() #Comentar pra inverter
    
#    pl.grid()
    
    # Deve-se criar as pastas './Resultados/Caracterizacao'
    pl.savefig('./imgs/Caracterizacao/percent_missing_data.eps', dpi=300)
    if show == True: pl.show()
    
    pl.close('all')
    
    return nan_inst

# In[]
    
def graficoDadosFaltantes(X):
    
    plotNames = renameNumericAtributes()
    
    nan_for_year = pd.DataFrame()
    for i, df in X.groupby(['year']):
        nan_for_year[i] = df.isna().sum()
    
    total_ano = nan_for_year.sum(axis=0)
    ptg = total_ano/total_ano.sum(axis=0)
    nan_for_year.loc['total_ano'] = total_ano
    
    total = nan_for_year.sum(axis=1)
    ptg = total/total.iloc[:-1].sum()
    ptg *= 100
    nan_for_year = pd.concat([nan_for_year, total.rename('total'), ptg.rename('%')], axis=1)
    
    nan_for_year = nan_for_year[nan_for_year['total'] != 0]
    nan_for_year = nan_for_year.drop('bank-size')
    
    ptg_ano = total_ano/nan_for_year.loc['total_ano', 'total']
    
    nan_for_year.loc['%'] = ptg_ano*100

    nan_for_year.iloc[:-1,:-1] = nan_for_year.iloc[:-1,:-1].applymap(lambda x: "{:d}".format(int(x)))
    
    pl.figure(3)
    nan_for_year.loc['%'][:-2].plot(kind='bar', color='blue', zorder=3)
    pl.grid(axis='y', zorder=0)
    pl.title('Total de dados faltantes por ano')
    pl.xticks(rotation=0)
    pl.yticks()
    pl.xlabel('Ano')
    pl.ylabel('Porcentagem (%)')
    pl.tight_layout()
    pl.savefig('./imgs/Caracterizacao/adados_faltantes_por_ano.eps', dpi=300)
    
    pl.figure(4)
#    pl.grid(True, axis='y', zorder=0)
    nan_for_year.rename(index=plotNames)['%'][:-2].sort_values().plot(kind='bar', color='blue', zorder=3)
    pl.grid(axis='y', zorder=0)
    pl.title('Dados faltantes por atributo numérico')
    pl.xlabel(r'Atributo')
    pl.ylabel('Porcentagem (%)')
    pl.xticks(rotation=45)
    pl.tight_layout()
    pl.savefig('./imgs/Caracterizacao/dados_faltantes_por_varaiavel.eps', dpi=300)

    pl.show()
    
    pl.close('all')
    
#    print('\n\nDADOS FALTANTES DE CADA COLUNA POR ANO\n')
#    print(nan_for_year.fillna('-'))    

    pl.close('all')    
    
    return nan_for_year

# In[]

def percentualDadosFaltantesPorAno(nan, op='f'):
    
    pl.figure()
    
    if op == 'f': 
        (nan.iloc[1:, :9].sum()/4256*100).plot(kind='bar', color='blue', zorder=3)
    else:
        (nan.iloc[0, :9]/4256*100).plot(kind='bar', color='blue', zorder=3)
        
    pl.grid(axis='y', zorder=0)

#    pl.yticks(range(0,101, 10))
    pl.xticks(rotation=0)
    
    if op == 'f':
        pl.title('Instâncias com dados faltantes')
    if op == 'c':
        pl.title('Instâncias com dados completos por ano')
        
    pl.xlabel(r'Ano')
    pl.ylabel('Porcentagem (%)')
    
    pl.tight_layout()
    
    if op == 'f':
        pl.savefig('./imgs/Caracterizacao/porcentagem_dados_faltantes_ano.eps', dpi=300)
    else:
        pl.savefig('./imgs/Caracterizacao/porcentagem_dados_completos_ano.eps', dpi=300)
        
    pl.show()
    pl.close('all')
    
# In[]
    
def percentualDadosPorBankSizeAno(X, op='f'):
    
    pl.figure()    
    sns.set_palette('Blues_d',len(X['bank-size'].unique()))
    
    nan = pd.concat([X.isna().sum(axis=1), X['year'], X['bank-size']], axis=1)
    
    if op == 'f':
        aux = nan[nan[0] != 0].groupby(['year', 'bank-size'])['bank-size'].count().unstack()
        esc = 'faltantes'
    else:
        aux = nan[nan[0] == 0].groupby(['year', 'bank-size'])['bank-size'].count().unstack()
        esc = 'completos'
        
    total = nan.groupby(['year', 'bank-size'])['bank-size'].count().unstack()
    
    ptg = (aux/total)*100
    ptg.columns.name = 'Bank Size'
    ptg.plot(kind='bar', zorder=3)
#    pl.legend(loc = 'upper right', bbox_to_anchor = (1.3, 1))
    pl.legend()
    pl.xticks(rotation=0)
    pl.xlabel('Ano')
    pl.ylabel('Porcentagem (%)')
    pl.grid(axis='y', zorder=0)
    pl.title('Percentual de dados ' + esc + ' por tamanho e ano')
    
    pl.savefig('./imgs/Caracterizacao/dados_por_bank-size' + esc +'.eps', dpi=300)    
    

# In[]
    
def histogramas(X, num_atributos, abrev, esc=''):
    
    pl.figure(1)
    X[num_atributos[:4]].rename(columns=abrev).hist(zorder=3, color='blue')
    pl.grid(axis='y', zorder=0)
    pl.savefig('./imgs/Caracterizacao/hists_pt1' + esc +'.eps', dpi=300)
    
    pl.figure(2)
    X[num_atributos[4:-1]].rename(columns=abrev).hist(zorder=3, color='blue')
    pl.grid(axis='y', zorder=0)
    pl.savefig('./imgs/Caracterizacao/hists_pt2' + esc +'.eps', dpi=300)
    
    pl.close('all')
    
    pl.figure(3)
    X[ num_atributos[-1] ].hist(zorder=3, color='blue',)
    pl.grid(axis='y', zorder=0)
    pl.title('Passivo', fontsize=22)
    pl.xticks(fontsize=20)
    pl.yticks(fontsize=20)
    pl.savefig('./imgs/Caracterizacao/hists_pt3' + esc +'.eps', dpi=300)
    pl.show()
    
    pl.close('all')
    
# In[]
    
def logaritmo(X,  num_atributos, abrev, plot = True):

    x = X.dropna().copy()
    
    x.loc[:, num_atributos] = np.log(x.loc[:, num_atributos]+1)
    
    scaler = StandardScaler().fit(x.loc[:, num_atributos])
    x.loc[:, num_atributos] = scaler.transform(x.loc[:, num_atributos])
    
    if plot == True: histogramas(x, num_atributos, abrev, esc='+log')
    
    return x
    
    
# In[]

def porcentagem(X, coluna):
    sns.set_palette('Blues_d',len(X[coluna].unique()))
    bar=X.groupby(['year',coluna])[coluna].count().unstack().fillna(0)
    bar.plot(kind='bar', stacked=False); 
    pl.legend(loc = 'upper right', bbox_to_anchor = (1.3, 1))
    pl.ylabel(u'Número de bancos')
    pl.xlabel(u'Year')
    
# In[]
    
def percentualDadosPorCCAno(X, coluna='country_code_cat', op='f'):
    
    pl.figure()    
    sns.set_palette('Blues_d',len(X[coluna].unique()))
    
    nan = pd.concat([X.isna().sum(axis=1), X['year'], X[coluna]], axis=1)
    
    if op == 'f':
        aux = nan[nan[0] != 0].groupby(['year', coluna])[coluna].count().unstack()
        esc = 'faltantes'
    else:
        aux = nan[nan[0] == 0].groupby(['year', coluna])[coluna].count().unstack()
        esc = 'completos'
        
    total = nan.groupby(['year', coluna])[coluna].count().unstack()
    
    ptg = (aux/total)*100
    ptg.columns.name = 'Bank Size'
    ptg.plot(kind='bar', zorder=3)
#    pl.legend(loc = 'upper right', bbox_to_anchor = (1.3, 1))
    pl.legend()
    pl.xticks(rotation=0)
    pl.xlabel('Ano')
    pl.ylabel('Porcentagem (%)')
    pl.grid(axis='y', zorder=0)
    pl.title('Percentual de dados ' + esc + ' por região e ano')
    
    pl.savefig('./imgs/Caracterizacao/dados_por_cc' + esc +'.eps', dpi=300)       
    
    
# In[]
    
def percentualDadosCat(X, show=False):
   
    tam = tamanho()
    loc = localizacao()
    
    pl.figure(1)
    x = X.groupby('country_code_cat')['country_code_cat'].count()*100/len(X)
    x.rename(loc).plot(kind='barh', color='blue', zorder=3)

    pl.xlabel('Porcentagem (%)')
#    pl.ylabel('Região')
    pl.ylabel('')
    pl.title('Porcentagem de dados por região')
    pl.grid(axis='x', zorder=0)
    pl.xticks(range(0, int(max(x)+5), 5))
    
    pl.tight_layout()
    pl.savefig('./imgs/Caracterizacao/percent_dados_por_regiao.eps', dpi=300)
    
    pl.figure(2)
    
    x = X.groupby('bank-size')['bank-size'].count()*100/len(X)
    x.rename(tam).plot(kind='barh', color='blue', zorder=3)
    
#    pl.xticks(tam)
    pl.xlabel('Porcentagem (%)')
#    pl.ylabel('Tamanho')
    pl.ylabel('')
    pl.title('Porcentagem de dados por tamanho do banco')
    pl.grid(axis='x', zorder=0)
    pl.xticks(range(0, int(max(x)+5), 5))
    
    pl.tight_layout()
    pl.savefig('./imgs/Caracterizacao/percent_dados_por_banksize.eps', dpi=300)

    pl.figure(3)
    
    x = X.groupby('specialisation_cat')['specialisation_cat'].count()*100/len(X)
    x.plot(kind='barh', color='blue', zorder=3)
    
    pl.xlabel('Porcentagem (%)')
#    pl.ylabel('Grupo de especialização')
    pl.ylabel('')
    pl.title('Porcentagem de dados por grupo de especialização')
    pl.grid(axis='x', zorder=0)
    pl.xticks(range(0, int(max(x)+5), 5))
    
    pl.tight_layout()
    pl.savefig('./imgs/Caracterizacao/percent_dados_por_specialisation.eps', dpi=300)
    
    pl.figure(4)
    x = X.groupby('consodilation_code')['consodilation_code'].count()*100/len(X)
    x.plot(kind='barh', color='blue', zorder=3)
    
    pl.xlabel('Porcentagem (%)')
#    pl.ylabel('Grupo de especialização')
    pl.ylabel('')
    pl.title('Porcentagem de dados por grupo de especialização')
    pl.grid(axis='x', zorder=0)
    pl.xticks(range(0, int(max(x)+5), 5))
    
    pl.tight_layout()
    pl.savefig('./imgs/Caracterizacao/percent_dados_por_consolidation.eps', dpi=300)

    if show == True: pl.show()
    
    pl.close('all')
    
# In[]
    
def boxplots(X, num_atributes):
    
    ren = renameNumericAtributes()
    x = logaritmo(X, num_atributes, ren, plot=False)    
    x.loc[:, num_atributes].rename(columns=ren).boxplot()
    
    pl.xticks(rotation=45)
    pl.title("Boxplot atributos numéricos")
    pl.tight_layout()
    pl.savefig("./imgs/Caracterizacao/pboxplots.png", dpi=300)
    
    pl.show()
    
    pl.close('all')
    
# In[]
    
def correlacoes(X, num_atributes, matrix = True, grafic = False):
    
    ren = renameNumericAtributes()
    x = logaritmo(X, num_atributes, ren, plot=False)   
    
    del ren['bank-size']
    
#    ren = dict(map(reversed, ren.items()))
    ren_array = np.array(sorted(ren.items(), key=lambda x: x[1]))
    
    pt_num_atributes = [i for i in ren_array if (i[0] in num_atributes) == True]
    pt_num_atributes = np.array(pt_num_atributes)

    
    if grafic == True:
        pl.rcParams.update(pl.rcParamsDefault)

        pl.figure()
        
    #    print(x.columns)
        sns.pairplot(x.dropna().rename(columns=ren), vars=pt_num_atributes[:,1], hue='bank-size')
        
    #    pl.xlabel()

        
        pl.savefig('./imgs/Caracterizacao/novacorrelacoes.eps', dpi=300)
    
    if matrix == True:
        pl.figure(figsize=(20,10))
#        pl.figure()
        sns.set(font_scale=1.4)
#        sns.heatmap(X_c[abrev_array[:,0]].dropna().corr(), xticklabels=abrev_array[:,1], yticklabels=abrev_array[:,1], linewidths=.5, annot=True)
        
        sns.heatmap(X[ren_array[:,0]].dropna().corr(), xticklabels=ren_array[:,1], yticklabels=ren_array[:,1], linewidths=.5, annot=True)
        
        locs, labels = pl.xticks()
        pl.setp(labels, rotation=0)
        
        pl.title('Matriz de correlação', fontsize=22)
        pl.tight_layout()
        
        pl.savefig('./imgs/Caracterizacao/matrizcorrelacoes.eps', dpi=350)
        
        pl.show()
    
    
    

# In[]   

def pcaIn(X, num_atributes, bin_atributes = []):
    
    ren = renameNumericAtributes()
    x = logaritmo(X, num_atributes, ren, plot=False)
    
#    if bin_atributes != []:
#        print('aqi')
#        x = pd.concat([x, X.dropna()[bin_atributes]], axis=1)
    
    del ren['bank-size']   
    
    x_an = x[num_atributes + bin_atributes]
    
    pca = PCA(n_components=4)
    # pca.fit(anlz.iloc[:,:4]).transform(anlz.iloc[:,:4])
    X_pca = pca.fit(x_an).transform(x_an)
    ## PCA process results
    results = pca.components_
    covm = pca.explained_variance_ratio_
    
    print(covm)
#    print(X_pca)
    
    vn1 = [(i, round(j, 4)) for i, j in zip(x_an.columns, results[0])]
    vn2 = [(i, round(j, 4)) for i, j in zip(x_an.columns, results[1])]
    # vn3 = [(i, j) for i, j in zip(T.iloc[:,:-2].columns, results[2])]
    
    vn1.sort(key=lambda x: -abs(x[1]))
    vn2.sort(key=lambda x: -abs(x[1]))
    # vn3.sort(key=lambda x: -abs(x[1]))
    
    print(vn1)
    print("\n", vn2)
    
#    print(len(pd.DataFrame(X_pca)), len(x[['bank-size', 'bank-size-index']]), 'jd')
    
#    print(pd.DataFrame(X_pca))
#    print(x[['bank-size', 'bank-size-index']])
    
#    pl.figure(figsize=(12,8))
    pl.figure()
    
    X_pca_scatter = pd.concat([pd.DataFrame(X_pca), x[['bank-size', 'bank-size-index']].reset_index(drop=True)], axis=1)
    
    colors_aux = ['r', 'y', 'g', 'b', 'pink', 'c', 'orange', 'silver', 'indigo', 'maroon']
    
    t = ['Pequeno', 'Médio', 'Grande', 'Super Grande']
    tam = dict(zip(range(5), t))
    
    for (i, j), df in X_pca_scatter.groupby(['bank-size-index', 'bank-size']):
        pl.scatter(df.iloc[:,0], df.iloc[:,1], marker='o', c=colors_aux[i], label=tam[i], s=25, edgecolor='k')
    
    pl.legend(title='Tamanho do banco:')
#    pl.xscale('log')
#    pl.yscale('log')
#    pl.tight_layout()
    pl.title('Resultado do ACP')
    pl.ylabel(r'$ACP \ Z_2$')
    pl.xlabel(r'$ACP \ Z_1$')
    #pl.tight_layout()
    pl.savefig('imgs/Caracterizacao/pcaInicial.png', dpi=600)
    pl.show()   

    return x

# In[]
    
def paresAtrubutosCat(X, show=False):
    
    tam = tamanho()
    reg = localizacao()

    pl.figure(1)
    sns.set_palette('Blues_d',len(X['country_code_cat'].unique()))
    x = X.groupby(['bank-size', 'country_code_cat'])['bank-size'].count().unstack().fillna(0)
    x.loc[:,:]=(x.values.T/x.sum(axis=1).values).T*100
    x.rename(index=tam, columns=reg).plot(kind='bar', zorder=3)
    pl.grid(axis='y', zorder=0)
    pl.xticks(rotation=0)
    pl.ylabel(u'Porcentagem (%)')
    pl.xlabel(u'Tamanho do Banco')
    pl.title('Quantidade de Bancos por Região e Tamanho')
    pl.legend(title='Região')
    pl.tight_layout()
    
    pl.savefig('./imgs/Caracterizacao/qtdRegTam.eps', dpi=500)    
    if show == True: pl.show()
    
    pl.figure(2)
    sns.set_palette('Blues_d',len(X['country_code_cat'].unique()))
    x = X.rename(tam).groupby(['specialisation_cat', 'country_code_cat'])['specialisation_cat'].count().unstack().fillna(0)
    x.loc[:,:]=(x.values.T/x.sum(axis=1).values).T*100
    x.rename(columns=reg).plot(kind='bar', zorder=3)
    pl.grid(axis='y', zorder=0)
    pl.xticks(rotation=0)
    pl.ylabel(u'Porcentagem (%)')
    pl.xlabel(u'Especialização do Banco')
    pl.title('Quantidade de Bancos por Região e Especialização')
    pl.legend(title='Região')
    pl.tight_layout()
    
    pl.savefig('./imgs/Caracterizacao/qtdRegEsp.eps', dpi=500)    
    if show == True: pl.show()
    
    pl.figure(3)
    sns.set_palette('Blues_d',len(X['bank-size'].unique()))
    x = X.rename(tam).groupby(['specialisation_cat', 'bank-size'])['specialisation_cat'].count().unstack().fillna(0)
    x.loc[:,:]=(x.values.T/x.sum(axis=1).values).T*100
    x.rename(columns=tam).plot(kind='bar', zorder=3)
    pl.grid(axis='y', zorder=0)
    pl.xticks(rotation=0)
    pl.ylabel(u'Porcentagem (%)')
    pl.xlabel(u'Especialização do Banco')
    pl.title('Quantidade de Bancos por Tamanho e Especialização')
    pl.legend(title='Tamanho do Banco')
    pl.tight_layout()
    
    pl.savefig('./imgs/Caracterizacao/qtdTamEsp.eps', dpi=500)    
    if show == True: pl.show()
    
    
    
    
    