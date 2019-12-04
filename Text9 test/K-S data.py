# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:25:11 2018

@author: chong
"""

from scipy.stats import ks_2samp
import numpy as np
import xlrd
import matplotlib.pyplot as plt

def read_data(st,sheetnum):
    tr_data=[]
    data=xlrd.open_workbook(st)
    table=data.sheets()[sheetnum]
    nrows=table.nrows
    ncols=table.ncols
    for i in range(1,nrows):
        tem=[]
        for j in range(ncols):
            tem.append(table.cell(i,j).value)     
        tr_data.append(tem)
    t_data=np.array(tr_data)
    t_data.reshape(nrows-1,ncols)
    #print(t_data.shape)
    return t_data
    

def _getdata():
    #training data
    #loc1
    str_data='./od/'+str(2)+'.xlsx'
    data0=read_data(str_data,0)
    
    #test data
    #loc2
    str_data='./od/'+str(7)+'.xlsx'
    data1=read_data(str_data,0)
    
    #test data
    #loc3
    str_data='./od/'+str(8)+'.xlsx'
    data2=read_data(str_data,0)
    
    #test data
    #loc4
    str_data='./od/'+str(5)+'.xlsx'
    data3=read_data(str_data,0)
    
    return data0,data1,data2,data3

def max_data(data1):
    amax=np.array([])
    [n,m]=data1.shape
    for i in range(m):
        tmax=data1[0][i]
        for j in range(n):
            if  data1[j][i]>tmax:
                tmax=data1[j][i]
        T=np.append(amax,tmax) 
        amax=T
    return amax
        
def min_data(data1):
    amin=np.array([])
    [n,m]=data1.shape
    for i in range(m):
        tmin=data1[0][i]
        for j in range(n):
            if data1[j][i]<tmin:
                tmin=data1[j][i]
        T=np.append(amin,tmin)
        amin=T
    return amin


def dis(maxdata,mindata,data):   
    
    ARR=[]
    PARR=[]
    for j in range(data.shape[1]):
        ar=np.linspace(mindata[j],maxdata[j],31)
        par=np.zeros((30))
        for item in data[:,j]:
            for i in range(31-1):
                if item>=ar[i] and item<ar[i+1]:
                    par[i]+=1
        ARR.append(ar)
        PARR.append(par)
        #print(par.shape,ar.shape)
        #plt.figure(figsize=(15*5,20*5))
        #plt.bar(range(len(par)),par)
    
    plt.figure(figsize=(15*5,20*5))
    ax1 = plt.subplot(4,1,1)
    ax2 = plt.subplot(4,1,2)
    ax3 = plt.subplot(4,1,3)
    ax4 = plt.subplot(4,1,4)
    ax1.bar(range(len(PARR[0])), PARR[0], tick_label=np.around(np.linspace(mindata[0],maxdata[0],30),decimals=1))
    ax2.bar(range(len(PARR[1])), PARR[1], tick_label=np.around(np.linspace(mindata[1],maxdata[1],30),decimals=1))
    ax3.bar(range(len(PARR[2])), PARR[2], tick_label=np.around(np.linspace(mindata[2],maxdata[2],30),decimals=1))
    ax4.bar(range(len(PARR[3])), PARR[3], tick_label=np.around(np.linspace(mindata[3],maxdata[3],30),decimals=1))



data0,data1,data2,data3=_getdata()
max0,min0=max_data(data0),min_data(data0)
max1,min1=max_data(data1),min_data(data1)
max2,min2=max_data(data2),min_data(data2)
max3,min3=max_data(data3),min_data(data3)

maxdata=[np.max([max0[0],max0[1],max0[2],max0[3]]),
         np.max([max1[0],max1[1],max1[2],max1[3]]),
         np.max([max2[0],max2[1],max2[2],max2[3]]),
         np.max([max3[0],max3[1],max3[2],max3[3]])]

mindata=[np.min([min0[0],min0[1],min0[2],min0[3]]),
         np.min([min1[0],min1[1],min1[2],min1[3]]),
         np.min([min2[0],min2[1],min2[2],min2[3]]),
         np.min([min3[0],min3[1],min3[2],min3[3]])]
mindata=[0.0,0.0,0.0,0.0]
print(maxdata,mindata)
data=[data0,data1,data2,data3]


dis(max3,min3,data3)

'''
for data in [data0,data1,data2,data3]:
    n,m=data0.shape
    print(n,m)
    for i in range(m):
        target=data0[:,i]
        K,p=ks_2samp(data[0:n,i],target)
        print(i,p,K)
    print("...................................")
'''

   