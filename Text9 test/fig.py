#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:59:42 2019

@author: chong
"""

import matplotlib.pyplot as plt
import numpy as np
import xlrd

'''
labels='frogs','hogs','dogs','logs'
sizes=15,20,45,10
colors='yellowgreen','gold','lightskyblue','lightcoral'
explode=0,0.1,0,0
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
plt.axis('equal')
plt.show()
'''

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

def training_fig():
    st='./final result/training_result.xlsx'
    data=read_data(st,0)
    t=range(data.shape[0])
    plt.plot(t,data[:,0],'r--',t,data[:,1],'b-.',t,data[:,2],'y:')
    plt.legend(['validation set','training set','test set'])
    plt.xlabel("Training steps")
    plt.ylabel("MSE")
    
def CKC_fig():
    st='./final result/CKCtest_data_training_result.xlsx'
    data0,data1,data2=read_data(st,0),read_data(st,1),read_data(st,2)
    t=range(data0.shape[0])
    plt.plot(t,data0[:,3],'r--',t,data1[:,3],'b-.',t,data2[:,3],'y:')
    plt.legend(['training set on location2','training set on location3','training set on location4'])
    plt.xlabel("Training steps")
    plt.ylabel("MSE")

def loc_test():
    #loc test
    st='./final result/loc_results.xlsx'
    data=read_data(st,2)
    t=range(data.shape[0])
    plt.figure(figsize=(15*5,20*5))
    ax1 = plt.subplot(4,1,1)
    ax2 = plt.subplot(4,1,2)
    ax3 = plt.subplot(4,1,3)
    ax4 = plt.subplot(4,1,4)
    ax1.plot(t,data[:,0],'r--',t,data[:,4],'b-.',t,data[:,8],'y:',t,data[:,12],'k-')
    ax2.plot(t,data[:,1],'r--',t,data[:,5],'b-.',t,data[:,9],'y:',t,data[:,13],'k-')
    ax3.plot(t,data[:,2],'r--',t,data[:,6],'b-.',t,data[:,10],'y:',t,data[:,14],'k-')
    ax4.plot(t,data[:,3],'r--',t,data[:,7],'b-.',t,data[:,11],'y:',t,data[:,15],'k-')
    
    ax1.set_ylabel("pH")
    ax2.set_ylabel("DO (mg/L)")
    ax3.set_ylabel("COD (mg/L)")
    ax4.set_ylabel("NH3-N (mg/L)")
    
    plt.legend(['TL model','Original model','Sup model','True'],loc = 'center left', bbox_to_anchor=(0.25, 4.9),ncol=4)
    plt.xlabel("Time (week)")
   
def time_test():
    #time test
    st='./final result/time_results.xlsx'
    data=read_data(st,0)
    t=range(data.shape[0])
    plt.figure(figsize=(15*5,20*5))
    ax1 = plt.subplot(4,1,1)
    ax2 = plt.subplot(4,1,2)
    ax3 = plt.subplot(4,1,3)
    ax4 = plt.subplot(4,1,4)
    ax1.plot(t,data[:,0],'r--',t,data[:,4],'b-.',t,data[:,8],'y:',t,data[:,12],'k-')
    ax2.plot(t,data[:,1],'r--',t,data[:,5],'b-.',t,data[:,9],'y:',t,data[:,13],'k-')
    ax3.plot(t,data[:,2],'r--',t,data[:,6],'b-.',t,data[:,10],'y:',t,data[:,14],'k-')
    ax4.plot(t,data[:,3],'r--',t,data[:,7],'b-.',t,data[:,11],'y:',t,data[:,15],'k-')
    
    ax1.set_ylabel("pH")
    ax2.set_ylabel("DO (mg/L)")
    ax3.set_ylabel("COD (mg/L)")
    ax4.set_ylabel("NH3-N (mg/L)")
    
    plt.legend(['TL model','Original model','Sup model','True'],loc = 'center left', bbox_to_anchor=(0.25, 4.9),ncol=4)
    plt.xlabel("Time (week)")

def mix_test():
    #mix test
    st='./final result/mix_results.xlsx'
    data=read_data(st,2)
    t=range(data.shape[0])
    plt.figure(figsize=(15*5,20*5))
    ax1 = plt.subplot(4,1,1)
    ax2 = plt.subplot(4,1,2)
    ax3 = plt.subplot(4,1,3)
    ax4 = plt.subplot(4,1,4)
    ax1.plot(t,data[:,0],'r--',t,data[:,4],'b-.',t,data[:,8],'y:',t,data[:,12],'k-')
    ax2.plot(t,data[:,1],'r--',t,data[:,5],'b-.',t,data[:,9],'y:',t,data[:,13],'k-')
    ax3.plot(t,data[:,2],'r--',t,data[:,6],'b-.',t,data[:,10],'y:',t,data[:,14],'k-')
    ax4.plot(t,data[:,3],'r--',t,data[:,7],'b-.',t,data[:,11],'y:',t,data[:,15],'k-')
    
    ax1.set_ylabel("pH")
    ax2.set_ylabel("DO (mg/L)")
    ax3.set_ylabel("COD (mg/L)")
    ax4.set_ylabel("NH3-N (mg/L)")
    
    plt.legend(['TL model','Original model','Sup model','True'],loc = 'center left', bbox_to_anchor=(0.25, 4.9),ncol=4)
    plt.xlabel("Time (week)")


#training_fig()
#CKC_fig()
#loc_test()
#time_test()
mix_test()
