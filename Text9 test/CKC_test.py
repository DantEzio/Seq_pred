#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:59:44 2019

@author: chong
"""
import tensorflow as tf
import numpy as np
import xlrd
import time
import pandas as pd


class wq_pred:
    def __init__(self):
        #get data
        self._getdata()
        
        self.in_size=self.data0.shape[1]
        #training_datain.shape[1]
        self.h_size=5#int(data0.shape[1]/3)
        self.out_size=self.data0.shape[1]
        #training_datain.shape[1]
        self.lr=0.001
        self.steps=50000
        self.lag=7
        
        
        #computer graph
        #3 models for original, traditional op, and TL
        self._build_model_1()
        self._build_model_2()
        self._build_model_3()
        #self.saver.save(self.sess, "Model/model.ckpt")
        self.sess=tf.compat.v1.Session()
    
    def _getdata(self):
        #training data
        #loc1
        str_data='./od/'+str(2)+'.xlsx'
        self.data0=self.read_data(str_data,0)
        self.outmax0=self.max_data(self.data0)
        self.outmin0=self.min_data(self.data0)
        self.t_data0=self.normalization_ver1(self.data0,self.outmin0,self.outmax0)
        
        #test data
        #loc2
        str_data='./od/'+str(7)+'.xlsx'
        self.data1=self.read_data(str_data,0)
        self.outmax1=self.max_data(self.data1)
        self.outmin1=self.min_data(self.data1)
        self.t_data1=self.normalization_ver1(self.data1,self.outmin1,self.outmax1)
        
        #test data
        #loc3
        str_data='./od/'+str(8)+'.xlsx'
        self.data2=self.read_data(str_data,0)
        self.outmax2=self.max_data(self.data2)
        self.outmin2=self.min_data(self.data2)
        self.t_data2=self.normalization_ver1(self.data2,self.outmin2,self.outmax2)
        
        #test data
        #loc4
        str_data='./od/'+str(5)+'.xlsx'
        self.data3=self.read_data(str_data,0)
        self.outmax3=self.max_data(self.data3)
        self.outmin3=self.min_data(self.data3)
        self.t_data3=self.normalization_ver1(self.data3,self.outmin3,self.outmax3)
    
    
    def max_data(self,data1):
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
        
    def min_data(self,data1):
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
    
    def normalization(self,data1):
        amin=self.min_data(data1)
        amax=self.max_data(data1)
        [n,m]=data1.shape
        for i in range(m):
            for j in range(n):
                if(amax[i]==amin[i]):
                    data1[j][i]=0
                else:
                    data1[j][i]=2*(data1[j][i]-amin[i])/(amax[i]-amin[i])-1
            
        return data1
    
    def normalization_ver1(self,data1,amin,amax):
        [n,m]=data1.shape
        for i in range(m):
            for j in range(n):
                data1[j][i]=2*(data1[j][i]-amin[i])/(0.001+amax[i]-amin[i])-1
            
        return data1
    
    def read_data(self,st,sheet):
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
    
        
    def _build_model_1(self):
        #model 1
        with tf.compat.v1.variable_scope('model1',reuse=tf.compat.v1.AUTO_REUSE):
            self.x_1=tf.compat.v1.placeholder(tf.float32,[None,self.in_size],name='x_1')
            self.y_1=tf.compat.v1.placeholder(tf.float32,[None,self.out_size],name='y_1')
            
            self.W11=tf.Variable(tf.ones([self.in_size,self.h_size]),name='m1w1')
            self.b11=tf.Variable(tf.ones([self.h_size]),name='m1b1')
            self.W21=tf.Variable(tf.ones([self.h_size,self.h_size]),name='m1w2')
            self.b21=tf.Variable(tf.ones([self.h_size]),name='m1b2')
            self.W31=tf.Variable(tf.ones([self.h_size,self.out_size]),name='m1w3')
            self.b31=tf.Variable(tf.ones([self.out_size]),name='m1b3')
                   
            self.h11=tf.nn.tanh(tf.matmul(self.x_1,self.W11)+self.b11)
            self.h21=tf.nn.tanh(tf.matmul(self.h11,self.W21)+self.b21)
    
            self.out_1=tf.nn.tanh(tf.matmul(self.h21,self.W31)+self.b31)
            
            self.loss1=tf.sqrt(tf.reduce_mean(tf.square((self.out_1 - self.y_1))))
            
            self.lossph=tf.sqrt(tf.reduce_mean(tf.square((self.out_1[0] - self.y_1[0]))))
            self.lossDO=tf.sqrt(tf.reduce_mean(tf.square((self.out_1[1] - self.y_1[1]))))
            self.lossCOD=tf.sqrt(tf.reduce_mean(tf.square((self.out_1[2] - self.y_1[2]))))
            self.lossNH=tf.sqrt(tf.reduce_mean(tf.square((self.out_1[3] - self.y_1[3]))))
            
            
            #self.loss1=tf.sqrt(tf.reduce_mean(tf.math.log(self.y_1/self.out_1)))
            self.train1=tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss1)
        
        
    def _build_model_2(self):
        #model 2
        #with tf.variable_scope('model2_old',reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope('model2',reuse=tf.compat.v1.AUTO_REUSE):
            self.x_2=tf.compat.v1.placeholder(tf.float32,[None,self.in_size],name='x_2')
            self.y_2=tf.compat.v1.placeholder(tf.float32,[None,self.out_size],name='y_2')
            
            self.W12=tf.Variable(tf.ones([self.in_size,self.h_size]),name='m2w1')
            self.b12=tf.Variable(tf.ones([self.h_size]),name='m2b1')
            self.W22=tf.Variable(tf.ones([self.h_size,self.h_size]),name='m2w2')
            self.b22=tf.Variable(tf.ones([self.h_size]),name='m2b2')
            self.W32=tf.Variable(tf.ones([self.h_size,self.out_size]),name='m2w3')
            self.b32=tf.Variable(tf.ones([self.out_size]),name='m2b3')
                   
            self.h12=tf.nn.tanh(tf.matmul(self.x_2,self.W12)+self.b12)
            self.h22=tf.nn.tanh(tf.matmul(self.h12,self.W22)+self.b22)
    
            self.out_2=tf.nn.tanh(tf.matmul(self.h22,self.W32)+self.b32)
            
            self.loss2=tf.sqrt(tf.reduce_mean(tf.square((self.out_2 - self.y_2))))
            #self.loss2=tf.sqrt(tf.reduce_mean(tf.math.log(self.y_2/self.out_2)))
            self.train2=tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss2)
        
        
    def _build_model_3(self):
        #model 3
        with tf.compat.v1.variable_scope('modelnew',reuse=tf.compat.v1.AUTO_REUSE):
            self.x_3=tf.compat.v1.placeholder(tf.float32,[None,self.in_size],name='x_3')
            self.y_3=tf.compat.v1.placeholder(tf.float32,[None,self.out_size],name='y_3')
            
            self.W13=tf.Variable(tf.ones([self.in_size,self.h_size]),name='m3w1')
            self.b13=tf.Variable(tf.ones([self.h_size]),name='m3b1')
            self.W23=tf.Variable(tf.ones([self.h_size,self.h_size]),name='m3w2')
            self.b23=tf.Variable(tf.ones([self.h_size]),name='m3b2')
            self.W33=tf.Variable(tf.ones([self.h_size,self.out_size]),name='m3w3')
            self.b33=tf.Variable(tf.ones([self.out_size]),name='m3b3')
                   
            self.h13=tf.nn.tanh(tf.matmul(self.x_3,self.W13)+self.b13)
            self.h23=tf.nn.tanh(tf.matmul(self.h13,self.W23)+self.b23)
    
            self.out_3=tf.nn.tanh(tf.matmul(self.h23,self.W33)+self.b33)
            
            self.loss3=tf.sqrt(tf.reduce_mean(tf.square((self.out_3 - self.y_3))))
            #self.loss3=tf.sqrt(tf.reduce_mean(tf.math.log(self.y_3/self.out_3)))
            self.train3=tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss3)
      
    def _train(self):
        print('start training')
        
        writer = pd.ExcelWriter('./CKCtest_data_training_result.xlsx')
        
        for i in [1,2,3]:
            #split data0
            if i ==1:
                [tnum,xnum]=self.t_data0.shape
                training_datain=self.t_data1[0:tnum,:]
                training_dataout=self.t_data1[1:tnum+1,:]
            elif i==2:
                [tnum,xnum]=self.t_data0.shape
                training_datain=self.t_data2[0:tnum,:]
                training_dataout=self.t_data2[1:tnum+1,:]
            else:
                [tnum,xnum]=self.t_data0.shape
                training_datain=self.t_data3[0:tnum,:]
                training_dataout=self.t_data3[1:tnum+1,:]
            
            #self.sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.train.Saver()
            self.sess=tf.Session()
            saver.restore(self.sess, "./Model/model.ckpt")
                    
            t_result=[]
            for t in range(self.steps):
                self.sess.run(self.train1,feed_dict={self.x_1:training_datain,self.y_1:training_dataout})
                l_t1=self.sess.run(self.lossph,feed_dict={self.x_1:training_datain,self.y_1:training_dataout})
                l_t2=self.sess.run(self.lossDO,feed_dict={self.x_1:training_datain,self.y_1:training_dataout})
                l_t3=self.sess.run(self.lossCOD,feed_dict={self.x_1:training_datain,self.y_1:training_dataout})
                l_t4=self.sess.run(self.lossNH,feed_dict={self.x_1:training_datain,self.y_1:training_dataout})
                
                t_result.append([l_t1,l_t2,l_t3,l_t4])
                
            
            csv_pd = pd.DataFrame(t_result) 
            csv_pd.to_excel(writer, sheet_name='data'+str(i),header=False,index = False)

if __name__=='__main__':
    wq_p=wq_pred()
    wq_p._train()

           