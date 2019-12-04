#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:54:37 2019

@author: chong
"""

import numpy as np
import xlrd
import pandas as pd
from scipy import stats

def read_data(st,sheet):
    tr_data=[]
    data=xlrd.open_workbook(st)
    table=data.sheets()[sheet]
    nrows=table.nrows
    ncols=table.ncols
    for i in range(nrows):
        tem=[]
        for j in range(ncols):
            s=table.cell(i,j).value
            if s!='':
                tem.append(float(s)) 
        tr_data.append(tem)
    t_data=np.array(tr_data)
    t_data.reshape(nrows,ncols)
    print(t_data.shape)
    return t_data

def save_xls_file(data,name): 
    csv_pd = pd.DataFrame(data)  
    csv_pd.to_csv(name+".csv", sep=',', header=False, index=False)


def MSE_BIC(truth,model,n,k):
    MSE=np.mean((truth-model)**2)
    print('MSE:',MSE)
    BIC=-2*np.log(MSE)+k*np.log(n)
    print('BIC:',BIC)


def R2(model,truth):
    def get_r2_numpy(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
        return np.sqrt(r_squared)
    r=get_r2_numpy(model,truth)
    print('r:',r)
    
def BF_AF(model,truth):
    BF=np.exp(np.mean(np.log(model/truth)))
    AF=np.exp(np.mean(np.abs(np.log(model/truth))))
    print('AF:',AF,'BF:',BF)
    
def p_value(model,truth):
    t, p= stats.ttest_ind(model,truth)
    print('t:',t,'p:',p)


#get data method
def loc_test():
    
    def show(pred,true,namemodel):
        print(namemodel)
        MSE_BIC(true,pred,true.shape[0],79)
        R2(true,pred)
        BF_AF(true,pred)
        p_value(true,pred)
       
    st='./final result/loc_results.xlsx'
    for loc_num in [0,1,2]:
        data=read_data(st,loc_num)
        phtruth,DOtruth,CODtruth,NH3truth=data[:,12],data[:,13],data[:,14],data[:,15]
        phmodelnew,DOmodelnew,CODmodelnew,NH3modelnew=data[:,0],data[:,1],data[:,2],data[:,3]
        phmodelold1,DOmodelold1,CODmodelold1,NH3modelold1=data[:,4],data[:,5],data[:,6],data[:,7]
        phmodelold2,DOmodelold2,CODmodelold2,NH3modelold2=data[:,8],data[:,9],data[:,10],data[:,11]
        print(loc_num,'loc_num','...................................')
        
        print('PH','###################################')
        show(phtruth,phmodelnew,'new')
        show(phtruth,phmodelold1,'old1')
        show(phtruth,phmodelold2,'old2')
        
        print('DO','###################################')
        show(DOtruth,DOmodelnew,'new')
        show(DOtruth,DOmodelold1,'old1')
        show(DOtruth,DOmodelold2,'old2')
        
        print('COD','###################################')
        show(CODtruth,CODmodelnew,'new')
        show(CODtruth,CODmodelold1,'old1')
        show(CODtruth,CODmodelold2,'old2')
        
        print('NH3','###################################')
        show(NH3truth,NH3modelnew,'new')
        show(NH3truth,NH3modelold1,'old1')
        show(NH3truth,NH3modelold2,'old2')
        
        
def time_test():
    
    def show(pred,true,namemodel):
        print(namemodel)
        MSE_BIC(true,pred,true.shape[0],79)
        R2(true,pred)
        BF_AF(true,pred)
        p_value(true,pred)
       
    st='./final result/time_results.xlsx'
    data=read_data(st,0)
    phtruth,DOtruth,CODtruth,NH3truth=data[:,12],data[:,13],data[:,14],data[:,15]
    phmodelnew,DOmodelnew,CODmodelnew,NH3modelnew=data[:,0],data[:,1],data[:,2],data[:,3]
    phmodelold1,DOmodelold1,CODmodelold1,NH3modelold1=data[:,4],data[:,5],data[:,6],data[:,7]
    phmodelold2,DOmodelold2,CODmodelold2,NH3modelold2=data[:,8],data[:,9],data[:,10],data[:,11]
        
    print('PH','###################################')
    show(phtruth,phmodelnew,'new')
    show(phtruth,phmodelold1,'old1')
    show(phtruth,phmodelold2,'old2')
    
    print('DO','###################################')
    show(DOtruth,DOmodelnew,'new')
    show(DOtruth,DOmodelold1,'old1')
    show(DOtruth,DOmodelold2,'old2')
    
    print('COD','###################################')
    show(CODtruth,CODmodelnew,'new')
    show(CODtruth,CODmodelold1,'old1')
    show(CODtruth,CODmodelold2,'old2')
    
    print('NH3','###################################')
    show(NH3truth,NH3modelnew,'new')
    show(NH3truth,NH3modelold1,'old1')
    show(NH3truth,NH3modelold2,'old2')
    
    
def mix_test():
    
    def show(pred,true,namemodel):
        print(namemodel)
        MSE_BIC(true,pred,true.shape[0],79)
        R2(true,pred)
        BF_AF(true,pred)
        p_value(true,pred)
       
    st='./final result/mix_results.xlsx'
    for loc_num in [0,1,2]:
        data=read_data(st,loc_num)
        phtruth,DOtruth,CODtruth,NH3truth=data[:,12],data[:,13],data[:,14],data[:,15]
        phmodelnew,DOmodelnew,CODmodelnew,NH3modelnew=data[:,0],data[:,1],data[:,2],data[:,3]
        phmodelold1,DOmodelold1,CODmodelold1,NH3modelold1=data[:,4],data[:,5],data[:,6],data[:,7]
        phmodelold2,DOmodelold2,CODmodelold2,NH3modelold2=data[:,8],data[:,9],data[:,10],data[:,11]
        print(loc_num,'loc_num','...................................')
        
        print('PH','###################################')
        show(phtruth,phmodelnew,'new')
        show(phtruth,phmodelold1,'old1')
        show(phtruth,phmodelold2,'old2')
        
        print('DO','###################################')
        show(DOtruth,DOmodelnew,'new')
        show(DOtruth,DOmodelold1,'old1')
        show(DOtruth,DOmodelold2,'old2')
        
        print('COD','###################################')
        show(CODtruth,CODmodelnew,'new')
        show(CODtruth,CODmodelold1,'old1')
        show(CODtruth,CODmodelold2,'old2')
        
        print('NH3','###################################')
        show(NH3truth,NH3modelnew,'new')
        show(NH3truth,NH3modelold1,'old1')
        show(NH3truth,NH3modelold2,'old2')

if __name__=='__main__':
    #loc_test()
    #time_test()
    mix_test()