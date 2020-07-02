#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:13:17 2019

@author: iagorosa
"""

import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
import scipy as sc

# In[]

#DEFINICOES
group_size = ['Small Bank', 'Mid-size Bank', 'Large Bank', 'Super Large Bank']
color = ['red', 'green', 'blue', 'yellow', 'purple', 'pink']
anos = [2007, 2011]
forms = ['^', 's']
reg_cat = ['center', 'periphery']

# Dicionario de bank-size:
# {0: 'Small Bank', 1: 'Mid-size Bank', 2: 'Large Bank', 3: 'Super Large Bank'}
#bs_dict = pd.Series(T['bank-size'].values, index = T['bank-size-index'].values).to_dict()

cs = ['b', 'orange', 'g', 'r']
# Associacao de cores ao bank-size-index:
# {0: 'r', 1: 'orange', 2: 'g', 3: 'b'}
#colors = dict(zip(T['bank-size-index'].unique(), [i for i in cs]))
colors = {0: 'r', 1: 'orange', 2: 'g', 3: 'b'}

# In[]

# Movimento anual dos bancos em relacao a media ou mediana, definidos pelo parametro 'agrupamento'
# Parametro coluna define o atributo para o groupby
def movimentoAnualBancosGeral(T, others = False, coluna = 'bank-size-index' , agrupamento = 'mean', show = False):
    
    colors = {0: 'r', 1: 'yellow', 2: 'g', 3: 'b'}

    if others == False:
        Taux = T[T['others'] == 0]
        esc = ''
    else:
        Taux = T
        esc = '+others'
        
    t = ['Pequeno', 'Médio', 'Grande', 'Super Grande']
    tam = dict(zip(range(5), t))
    
    cl = 'Cluster ' if coluna == 'cluster' else ''
    pl.figure(figsize=(8,8))    
    for k,(y, df) in enumerate(Taux.groupby(coluna)):
            m=df.groupby('year').agg(agrupamento)[['AL', 'PL']]
                        
#             Descomentar caso queira o desvio padrao (grafico fica muito feio)
#            sv = df.groupby('year').std()[['AL', 'PL']]
#            pl.errorbar(m['AL'], m['PL'], xerr = sv['AL'], marker='o', label=y, color=color[k])
            
            pl.plot(m['AL'], m['PL'], 'o-', label=cl+str(tam[y]), color=colors[k]); #pl.title(y)
            for i in range(len(m)):
                a,b,s = m['AL'].iloc[i],m['PL'].iloc[i], m.index[i]
                pl.text(a,b,s)
    
    pl.title('Movimento dos bancos ao longo dos anos', fontsize=14)
    pl.xlabel('Liquidez do Ativo', fontsize=12);pl.ylabel('Liquidez do passivo', fontsize=12)
    pl.legend(title='Tamanho do banco')
    pl.grid()
    pl.savefig('imgs/movimento_anual' + esc + '.eps', dpi=300)
    if show == True: pl.show()
    pl.close('all')
    
# In[]
    
# Movimento anual dos bancos em relacao a media ou mediana, definidos pelo parametro 'agrupamento'
# Usando definicao regional dos bancos 
# Parametro coluna define o atributo para o groupby
def movimentoAnualBancosRegional(T, others: bool = False, coluna = 'bank-size-index' , agrupamento = 'mean', show = False):
    
    t = ['Banco Pequeno', 'Banco Médio', 'Banco Grande', 'Banco Super Grande']
    tam = dict(zip(range(5), t))
    
    reg_name = {'periphery': 'Periferia', 'center': 'Centro', 'others': 'Outros'}

    
    if others == False:
        reg_cat = ['center', 'periphery']
        esc = ''
    else:
        reg_cat = ['center', 'periphery', 'others']
        esc = '+others'
        
#    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
#                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
#                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
#                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
                  
    color_sequence = ['#d62728', '#ff9896', '#FFFF00', '#fbec5d',
                      '#2ca02c', '#98df8a', '#1f77b4', '#aec7e8']
    
    cl = 'Cluster ' if coluna == 'cluster' else ''
    j = 0
        
    # Gera um grafico para cada tipo de 'coluna' em relação ao movimento dos bancos
    for y, df in T.groupby(coluna):
        for k, reg in enumerate(reg_cat):
            df_aux = df[df[reg] == 1]
            m=df_aux.groupby('year').agg(agrupamento)[['AL', 'PL']]
            
            # Descomentar caso queira o desvio padrao (grafico fica muito feio)
#            sv = df_aux.groupby('year').std()[['AL', 'PL']]
#            pl.errorbar(m['AL'], m['PL'], xerr = sv['AL'], marker='o', label=y, color=color[k])
            mark = '-o' if reg == 'center' else '-^'
            pl.figure(1, figsize=(8,8))
#            pl.plot(m['AL'], m['PL'], 'o-', label=cl+str(y)+'\n'+reg, color=color_sequence[j]); j+= 1
            pl.plot(m['AL'], m['PL'], mark, label=cl+str(tam[y])+'\n'+reg_name[reg], color=color_sequence[j]); j+= 1
            
#            pl.figure(2, figsize=(8,8))
#            pl.plot(m['AL'], m['PL'], 'o-', label=reg, color=color[k])
            
            for i in range(len(m)):
                a,b,s = m['AL'].iloc[i],m['PL'].iloc[i], m.index[i]
                pl.figure(1)
                pl.text(a,b,s)
#                pl.figure(2)
#                pl.text(a,b,s)
        
#        pl.figure(1)
        pl.xlabel('Liquidez do Ativo', fontsize=12);pl.ylabel('Liquidez do Passivo', fontsize=12)
        pl.legend()
        pl.title('Movimento dos Bancos - ' + cl+str(tam[y]), fontsize = 14)
        pl.grid()
        pl.savefig('imgs/movimento_anual' + esc + ' (' + cl+str(tam[y]) + ').eps')
        if show == True: pl.show()
        pl.close()
        
    # Grafico geral do movimento dos bancos. Inclui todos os tipo de bancos em 'coluna'
    for j, (y, df) in enumerate(T.groupby(coluna)):
        pl.figure(2, figsize=(10,8))    
        for k, reg in enumerate(reg_cat):
            df_aux = df[df[reg] == 1]
            m=df_aux.groupby('year').agg(agrupamento)[['AL', 'PL']]
            
            mark = '-o' if reg == 'center' else '-^'
            pl.plot(m['AL'], m['PL'], mark, label=cl+str(tam[y])+'\n'+reg_name[reg], color=color_sequence[2*j+k])
            
            for i in range(len(m)):
                a,b,s = m['AL'].iloc[i],m['PL'].iloc[i], m.index[i]
                pl.text(a,b,s)    
        
    pl.xlabel('Liquidez do Ativo', fontsize=12);pl.ylabel('Liquidez do Passivo', fontsize=12)
    pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pl.title('Movimento dos Bancos por Região ', fontsize = 14)
    pl.grid()
    pl.tight_layout()
    pl.savefig('imgs/[REGIONAL] movimento_anual_geral' + esc + '.eps')
    if show == True: pl.show()
    
    pl.close('all')
    
# In[]

# Graficos PDFs para a liquidez do ativo e do passivo
# PDF utilizada: Kernel Density Estimation (KDE)
def densidadeDeProbabilidadeGeral(T, others = False, show = False, coluna = 'bank-size-index'):  
    
    t = ['Pequeno', 'Médio', 'Grande', 'Super Grande']
    tam = dict(zip(range(5), t))

    colors = {0: 'r', 1: 'y', 2: 'g', 3: 'b'}
    cc = pd.Series(colors)
    ccdf = pd.DataFrame(cc, columns=['color'])

    if others == False:
        Taux = T[T['others'] == 0]
        esc = ''
    else:
        Taux = T
        esc = '+others'
    
    if coluna == 'cluster':
        bs = [(i, 'Cluster '+str(i)) for i in T['cluster'].unique()]
        bs_dict = dict(bs)
    else:
        bs_dict = pd.Series(T['bank-size'].values, index = T['bank-size-index'].values).to_dict()

    ## KDEPLOT APENAS SEPARADOS POR ANO, ['AL', 'PL'] E TAMANHO DOS BANCOS (ou cluster)
    for y, df in Taux.groupby('year'):
#        classnames, indices = np.unique(group_size, return_inverse=True)
        pl.figure(figsize=(8,8))     
        for v in ['AL', 'PL']:
            nome = 'Liquidez do Ativo' if v == 'AL' else 'Liquidez do Passivo'
            pl.cla()
            for x,df1 in df.groupby(coluna):
#                sns.kdeplot(df1[v],  label=bs_dict[x], shade=1,lw=2, color=colors[x])
                sns.kdeplot(df1[v],  label=tam[x], shade=1,lw=2, color=colors[x])
            
            pl.title(str(y)+' -- '+nome)
            pl.legend(title='Tamanho do banco')
            pl.xlabel(v, fontsize = 12)
#            pl.ylabel(r'$P[X \leq x]$', fontsize = 12)
            pl.tight_layout()
            pl.savefig('imgs/[GERAL] Probability Density ' + esc + '- ' + v +'-'+str(y)+'.png', depi=500)
            if show == True: pl.show()     
        pl.close('all')
    
        pl.gcf().set_size_inches(16, 16)
        
        # Grafico scatter no plano ALxPL + kdeplot nos eixos 
        df = pd.merge(ccdf, df, how='inner', left_index=True, right_on=coluna)
        g = sns.JointGrid(x='AL', y='PL', data=df, ratio=2)
        g = g.plot_joint(pl.scatter, c=df['color'], edgecolor="black")
        for c in sorted(df[coluna].unique()):
            sns.kdeplot(df[df[coluna]==c]['PL'], ax=g.ax_marg_y, vertical=True,
                         shade=True, label = list(tam.values())[c], color=colors[c], legend=False)
#            g
            g2 = sns.kdeplot(df[df[coluna]==c]['AL'], ax=g.ax_marg_x, vertical=False,
                        shade=True, label = list(tam.values())[c], color=colors[c], legend=True)
            g2.legend(title='Tamanho do banco')
        
        pl.title(str(y), fontsize = 18)
        pl.xlabel("LA"); pl.ylabel('LP')
        pl.tight_layout()
        pl.savefig('imgs/[GERAL] Probability Density + Scatter ' + esc + '- '+ str(y)+'.png', dpi = 500)
        if show == True: pl.show()     
        
        
        pl.close('all')
        
        # Grafico kldplot 2D, ou seja, usando ambas variaveis (AL, PL)
        
#        for c in sorted(df['bank-size-index'].unique()):
#            sns.kdeplot(df[df['bank-size-index']==c]['AL'], df[df['bank-size-index']==c]['PL'], cmap="Blues", shade=True, shade_lowest=False)
#        
#            pl.title(str(y) + ' -- ' + bs_dict[c], fontsize = 18)
#            pl.tight_layout()
#            if show == True: pl.show()     
#            pl.savefig('imgs/[GERAL] Bivariate kde plot-' + bs_dict[c] + esc + '- '+ str(y)+'.png', dpi = 500)
            
    pl.close('all')
    
# In[]
    
    # Graficos PDFs para a liquidez do ativo e do passivo, por cada regiao
# PDF utilizada: Kernel Density Estimation (KDE)
def densidadeDeProbabilidadeRegional(T, others = False, show = False, coluna = 'bank-size-index'):

    colors = {0: 'r', 1: 'y', 2: 'g', 3: 'b'}
    cc = pd.Series(colors)
    ccdf = pd.DataFrame(cc, columns=['color'])

    t = ['Pequeno', 'Médio', 'Grande', 'Super Grande']
    tam = dict(zip(range(5), t))    
    reg_name = {'periphery': 'Periferia', 'center': 'Centro', 'others': 'Outros'}
    
    if coluna == 'cluster':
        bs = [(i, 'Cluster '+str(i)) for i in T['cluster'].unique()]
        bs_dict = dict(bs)
    else:
        bs_dict = pd.Series(T['bank-size'].values, index = T['bank-size-index'].values).to_dict()
        
    for y, df in T.groupby('year'):
#        classnames, indices = np.unique(group_size, return_inverse=True)    
        for v in ['AL', 'PL']:
            nome = 'Asset Liquidity' if v == 'AL' else 'Liability Liquidity'
            for reg in reg_cat:
#                pl.cla()
                pl.figure(figsize=(8,8)) 
                df_aux = df[df[reg] == 1]
                for x,df1 in df_aux.groupby(coluna):
                        sns.kdeplot(df1[v],  label=tam[x], shade=1,lw=2, color=colors[x])
                
                pl.title(str(y)+' -- '+ nome + ' (' + reg_name[reg] + ')')
                pl.legend(title='Tamanho do banco')
                pl.xlabel(v)
                pl.tight_layout()  
                pl.savefig('imgs/[REGIONAL] ' + reg + ' - RegionDensityProb_' + v +'-'+str(y)+'.png', dpi=500)
                if show == True: pl.show()    
        
                pl.close('all') 
        pl.gcf().set_size_inches(20, 10)
        for reg in reg_cat:
    #        pl.cla()
            df_aux = df[df[reg] == 1]
            df_aux = pd.merge(ccdf, df_aux, how='inner', left_index=True, right_on=coluna)
            g = sns.JointGrid(x='AL', y='PL', data=df_aux, ratio=2)
            g = g.plot_joint(pl.scatter, c=df_aux['color'], edgecolor="black")#, cmap=cmap)
            for c in sorted(df[coluna].unique()):
                sns.kdeplot(df_aux['PL'][df_aux[coluna]==c], ax=g.ax_marg_y, vertical=True,
                             shade=True, label = list(tam.values())[c], color=colors[c], legend=False)
                g2=sns.kdeplot(df_aux['AL'][df_aux[coluna]==c], ax=g.ax_marg_x, vertical=False,
                            shade=True, label = list(tam.values())[c], color=colors[c], legend=True)
            
            g2.legend(title='Tamanho do banco')
            pl.title(str(y) + ' -- ' + reg_name[reg], fontsize = 14)
            pl.xlabel("LA"); pl.ylabel('LP')
            pl.tight_layout()
            pl.savefig('imgs/[REGIONAL] ' + reg +  ' - Probability Density + Scatter - ' + '_'+str(y)+'.png', dpi=500)
            if show == True: pl.show()
        
        pl.close('all')
        
# In[]

# Funcao auxiliar para calculo de distancia entre classes
def distancias(T, coluna, reg = None, others = False):
    
    anos = list(sorted(T['year'].unique()))
    dist = np.zeros((len(anos), 4, 4))
    
    if others == False:
        Taux = T[T['others'] == 0]
    else:
        Taux = T
            
    for y, df in Taux.groupby('year'):
        
        if reg == None:
            centroides =  df.groupby(coluna, sort=True).mean()[['AL', 'PL']].values
        else:
            centroides =  df[df[reg] == 1].groupby(coluna).mean()[['AL', 'PL']].values
            
        dist[anos.index(y), :, :] = sc.spatial.distance.cdist(centroides, centroides, 'euclidean')    
    
    return dist  

# In[]

# Grafico da evolucao da disatancia entre classes 
def graficoDistanciasGeral(T, others = True, show = True, coluna = 'bank-size-index'):
    
    d = distancias(T, coluna)
    
    t = ['Pequeno', 'Médio', 'Grande', 'Super Grande']
    tam = dict(zip(range(5), t)) 
    colors = {0: 'r', 1: 'y', 2: 'g', 3: 'b'}
    
    if others == False:
        Taux = T[T['others'] == 0]
        esc = ''
    else:
        Taux = T
        esc = '+others'
        
    if coluna == 'cluster':
        bs = [(i, 'Cluster '+str(i)) for i in T['cluster'].unique()]
        bs_dict = dict(bs)
    else:
        bs_dict = pd.Series(T['bank-size'].values, index = T['bank-size-index'].values).to_dict()
    
    pl.figure(figsize=(12,8))    
    ano = sorted(Taux['year'].unique())
#    for i, a in enumerate(range(ano[0], ano[-1])):
    for i in range(4):
        for j in range(4):
            
            if j > i:
            
                pl.plot(d[:, i, j] , 'o-', label='d('+str(tam[i])+', '+str(tam[j])+')' )
    
    pl.xticks(range(0, len(d)), range(ano[0], ano[-1]+1))
    pl.title('Distância entre grupos ao longo dos anos', fontsize=16)
    pl.xlabel('Ano', fontsize=14);pl.ylabel(r'$||u - v||_2$', fontsize=14)
    pl.legend(title='Distância entre os bancos')
    pl.tight_layout()
    pl.grid()
    pl.savefig('imgs/distancia_grupos' + esc  + '.eps', dpi=500)
    if show == True: pl.show()
    pl.close('all')
    
# In[]
    
def graficoDistanciasRegional(T, others = False, show = True, coluna = 'bank-size-index'):
    
    if others == False:
        reg_cat = ['center', 'periphery']
        esc = ''
    else:
        reg_cat = ['center', 'periphery', 'others']
        esc = '+others'
        
    if coluna == 'cluster':
        bs = [(i, 'Cluster '+str(i)) for i in T['cluster'].unique()]
        bs_dict = dict(bs)
    else:
        bs_dict = pd.Series(T['bank-size'].values, index = T['bank-size-index'].values).to_dict()
    
    for reg in reg_cat:
        d = distancias(T, coluna, reg = reg)
        
        pl.figure(figsize=(12,8))    
        ano = sorted(T['year'].unique())
    #    for i, a in enumerate(range(ano[0], ano[-1])):
        for i in range(4):
            for j in range(4):
                
                if j > i:
                
    #                pl.plot(d[:, i, j] , 'o-', label='d('+str(i)+','+str(j)+')' )
                    pl.plot(d[:, i, j] , 'o-', label='d('+str(bs_dict[i])+', '+str(bs_dict[j])+')' )
        
        pl.xticks(range(0, len(d)), range(ano[0], ano[-1]+1))
        pl.title('Distância entre grupos ao longo dos anos - ' + reg, fontsize=14)
        pl.xlabel('Ano', fontsize=12);pl.ylabel(r'$||u - v||_2$', fontsize=12)
        pl.legend()
#        pl.grid()
        pl.tight_layout()
        pl.savefig('imgs/distancia_grupos' + esc + '-' + reg+  '.png', dpi=500)
        if show == True: pl.show()
        pl.close('all')
        