#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 04:33:33 2019

@author: iagorosa
"""

## error bar parametro capsize = 4

import pandas as pd
import pylab as pl
import seaborn as sns

# In[]

def histogramas_crise(X, atributos, ren_atr, esc=''):
    
    pl.figure()
    
    for i, at in enumerate(atributos):
    
        ax=X[at].rename(columns=ren_atr).hist(zorder=3, color='blue', bins=20)
        X[at].rename(columns=ren_atr).plot(kind='kde', ax=ax, color='red')
        
        pl.grid(axis='y', zorder=0)
        pl.title(list(ren_atr.items())[i][1])
        
        pl.savefig('./imgs/crise/hists_crise_' + at + '_' + str(i) + esc +'.eps', dpi=300)
        pl.show()
        
        pl.close()
        
        ax=sns.distplot(X[at])
        pl.setp(ax.lines, zorder=100)
        pl.grid(axis='y', zorder=0)
        pl.title('Distribuição da '+ ren_atr[at])
        xt = 'LA' if at == 'AL' else 'LP'
        pl.xlabel(xt)
        pl.savefig('./imgs/crise/hists_kde_crise_' + at + '_' + str(i) + esc +'.png', dpi=300)
        pl.show()
        
# In[]

def pref_liquidez(X, atr):
    
    t = ['Pequeno', 'Médio', 'Grande', 'Super Grande']
    tam = dict(zip(range(5), t))
    new_names = {'DPA': 'PDA', 'CPA': 'PEA', 'LTL': 'PLP', 'STL': 'PCP'}
    
    X_c = X.copy()

    X_c.rename(columns=new_names, inplace=True)

    for i, df in X_c.groupby(['bank-size-index']):

        pl.figure()
        df.groupby('year')[['PDA', 'PEA']].agg('mean').plot(marker='o')
        pl.xlabel('Ano')
        pl.title('Composição da Liquidez do Ativo -- '+'Banco '+tam[i]) 
        pl.grid()
        pl.tight_layout()
        pl.savefig('./imgs/crise/comp_LA_' + tam[i] +'.eps', dpi=300)
        
        pl.figure()
        df.groupby('year')[['PCP', 'PLP']].agg('mean').plot(marker='o')
        pl.xlabel('Ano')
        pl.title('Composição da Liquidez do Passivo -- '+'Banco '+tam[i]) 
        pl.grid()
        pl.tight_layout()
        pl.savefig('./imgs/crise/comp_LP_' + tam[i] + '.eps', dpi=300)
        
# In[]
        
def pref_liquidez_reg(X, atr):
    
    t = ['Pequeno', 'Médio', 'Grande', 'Super Grande']
    tam = dict(zip(range(5), t))
    
    reg = {'periphery': 'Periferia', 'center': 'Centro', 'others': 'Outros'}
    new_names = {'DPA': 'PDA', 'CPA': 'PEA', 'LTL': 'PCP', 'STL': 'PLP'}
    
    X_c = X.copy()

    X_c.rename(columns=new_names, inplace=True)

    for (i, j), df in X_c.groupby(['bank-size-index', 'country_code_cat']):

        pl.figure()
        df.groupby('year')[['PDA', 'PEA']].agg('mean').plot(marker='o')
        pl.xlabel('Ano')
        pl.title('Composição da Liquidez do Ativo'+'\nBanco '+tam[i] + " -- "+ reg[j]) 
        pl.grid()
        pl.tight_layout()
        pl.savefig('./imgs/crise/[REGIONAL]comp_LA_' + tam[i]+reg[j] +'.eps', dpi=300)
        
        pl.figure()
        df.groupby('year')[['PCP', 'PLP']].agg('mean').plot(marker='o')
        pl.xlabel('Ano')
        pl.title('Composição da Liquidez do Passivo'+'\nBanco '+tam[i]+ " -- "+reg[j]) 
        pl.grid()
        pl.tight_layout()
        pl.savefig('./imgs/crise/[REGIONAL]comp_LP_' + tam[i]+reg[j] + '.eps', dpi=300)
        
#        pl.show()
        pl.close('all')



