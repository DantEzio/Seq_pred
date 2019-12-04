#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 22:36:17 2019

@author: chong
"""
import tensorflow as tf
import numpy as np
import xlrd
import time
import pandas as pd

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

def normalization(data1):
    amin=min_data(data1)
    amax=max_data(data1)
    [n,m]=data1.shape
    for i in range(m):
        for j in range(n):
            if(amax[i]==amin[i]):
                data1[j][i]=0
            else:
                data1[j][i]=2*(data1[j][i]-amin[i])/(amax[i]-amin[i])-1
        
    return data1

def normalization_ver1(data1,amin,amax):
    [n,m]=data1.shape
    for i in range(m):
        for j in range(n):
            data1[j][i]=2*(data1[j][i]-amin[i])/(0.001+amax[i]-amin[i])-1
        
    return data1

def read_data(st,sheet):
    tr_data=[]
    data=xlrd.open_workbook(st)
    table=data.sheets()[sheet]
    nrows=table.nrows
    ncols=table.ncols
    for i in range(1,nrows):
        tem=[]
        for j in range(ncols):
            s=table.cell(i,j).value
            tem.append(float(s)) 
        tr_data.append(tem)
    t_data=np.array(tr_data)
    t_data.reshape(nrows-1,ncols)
    print(t_data.shape)
    return t_data

def save_xls_file(data,name): 
    csv_pd = pd.DataFrame(data)  
    csv_pd.to_csv(name+".csv", sep=',', header=False, index=False)

for num in range(1):
    #training data
    str_data='./od/'+str(num)+'.xlsx'
    print(str_data)
    data0=read_data(str_data,0)
    
    
    outmax0=max_data(data0)
    outmin0=min_data(data0)
    t_data0=normalization_ver1(data0,outmin0,outmax0)
    [tnum,xnum]=t_data0.shape
    
    training_num=int(2*tnum/3)
    
    training_datain=t_data0[0:training_num-1,:]
    training_dataout=t_data0[1:training_num,:]
    
    #test data
    test_datain1=t_data0[0:tnum-1,:]
    data0=read_data(str_data,0)
    test_dataout1=data0[1:tnum,:]
    
    print(len(training_datain),len(training_dataout))
    print(len(test_datain1),len(test_dataout1))
    
    #computer graph
    in_size=data0.shape[1]
    #training_datain.shape[1]
    h1_size=5#int(data0.shape[1]/3)
    h2_size=5#int(data0.shape[1]/3)
    h3_size=20#int(data0.shape[1]/3)
    h4_size=15#int(data0.shape[1]/3)
    out_size=data0.shape[1]
    #training_datain.shape[1]
    lr=0.01
    steps=1000
    batch_size=100
    
    x_=tf.placeholder(tf.float32,[None,in_size])
    y_=tf.placeholder(tf.float32,[None,out_size])
    
    #W1=tf.Variable(tf.truncated_normal([in_size,h1_size],stddev=0.1))
    W1=tf.Variable(tf.ones([in_size,h1_size]))
    b1=tf.Variable(tf.ones([h1_size]))
    
    #W2=tf.Variable(tf.truncated_normal([h1_size,h2_size],stddev=0.1))
    W2=tf.Variable(tf.ones([h1_size,h2_size]))
    b2=tf.Variable(tf.ones([h2_size]))
    
    W3=tf.Variable(tf.ones([h2_size,out_size]))
    b3=tf.Variable(tf.ones([out_size]))
    
    W4=tf.Variable(tf.ones([h3_size,out_size]))
    b4=tf.Variable(tf.ones([out_size]))
    
    #W5=tf.Variable(tf.ones([h4_size,out_size]))
    #b5=tf.Variable(tf.ones([out_size]))
    
    
    h1=tf.nn.tanh(tf.matmul(x_,W1)+b1)
    h2=tf.nn.tanh(tf.matmul(h1,W2)+b2)
    #h3=tf.nn.tanh(tf.matmul(h2,W3)+b3)
    #h4=tf.nn.tanh(tf.matmul(h3,W4)+b4)
    out_=tf.nn.tanh(tf.matmul(h2,W3)+b3)
    
    loss=tf.sqrt(tf.reduce_mean(tf.square(out_ - y_)))
    train=tf.train.AdamOptimizer(lr).minimize(loss)
    
    saver = tf.train.Saver()  
    
    print('start training')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t in range(steps):     
            for i in range(100):
                t_temx=training_datain[i,:]
                t_temy=training_datain[i+1,:]
                bx=np.array([t_temx])
                by=np.array([t_temy])
                #bx.reshape([None,in_size])
                #by.reshape([None,out_size])
    
                sess.run(train,feed_dict={x_:bx,y_:by})
                
        #saver.save(sess, "Model_ANNseq/model.ckpt")
     
        #element0
        starttime = time.time()
        output0=[]
        for i in range(7,test_datain1.shape[0]):
            batch_xs=np.array([test_datain1[i,:]])
            batch_xs.reshape([1,in_size])
            pre=sess.run(out_,feed_dict={x_:batch_xs})      
            result=((pre+1)/2)*(0.001+outmax0-outmin0)+outmin0
            output0.append([result]) 
        result0=np.array([output0])
        result0=result0.reshape(test_datain1.shape[0]-7,test_datain1.shape[1])
        endtime = time.time()
        print(endtime - starttime)
        #np.savetxt("ANN_test.txt", result1)
        save_xls_file(result0,'./result/'+str(num)+'_old')
