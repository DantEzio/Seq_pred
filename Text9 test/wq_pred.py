#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 21:39:12 2019

@author: chong
"""

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
        self.steps=5000
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
        #split data0
        a=0.4
        [tnum,xnum]=self.t_data0.shape
              
        training_datain=self.t_data0[0:int(a*tnum),:]
        training_dataout=self.t_data0[1:int(a*tnum)+1,:]
        
        validation_datain=self.t_data0[int(a*0.6*tnum):int(a*0.8*tnum),:]
        validation_dataout=self.t_data0[int(a*0.6*tnum)+1:int(a*0.8*tnum)+1,:]
        
        test_datain=self.t_data0[int(a*0.8*tnum):int(a*tnum)-2,:]
        test_dataout=self.t_data0[int(a*0.8*tnum)+1:int(a*tnum)-1,:]
        
        self.sess.run(tf.compat.v1.global_variables_initializer())
        #saver = tf.train.Saver()
        #self.sess=tf.Session()
        #saver.restore(self.sess, "./Model/model.ckpt")
                
        t_result=[]
        start =time.clock()
        for t in range(self.steps):
            self.sess.run(self.train1,feed_dict={self.x_1:training_datain,self.y_1:training_dataout})
            l_t1=self.sess.run(self.loss1,feed_dict={self.x_1:training_datain,self.y_1:training_dataout})
            l_v1=self.sess.run(self.loss1,feed_dict={self.x_1:validation_datain,self.y_1:validation_dataout})
            l_test1=self.sess.run(self.loss1,feed_dict={self.x_1:test_datain,self.y_1:test_dataout})
            
            self.sess.run(self.train2,feed_dict={self.x_2:training_datain,self.y_2:training_dataout})
            l_t2=self.sess.run(self.loss2,feed_dict={self.x_2:training_datain,self.y_2:training_dataout})
            l_v2=self.sess.run(self.loss2,feed_dict={self.x_2:validation_datain,self.y_2:validation_dataout})
            l_test2=self.sess.run(self.loss2,feed_dict={self.x_2:test_datain,self.y_2:test_dataout})
            
            self.sess.run(self.train3,feed_dict={self.x_3:training_datain,self.y_3:training_dataout})
            l_t3=self.sess.run(self.loss3,feed_dict={self.x_3:training_datain,self.y_3:training_dataout})
            l_v3=self.sess.run(self.loss3,feed_dict={self.x_3:validation_datain,self.y_3:validation_dataout})
            l_test3=self.sess.run(self.loss3,feed_dict={self.x_3:test_datain,self.y_3:test_dataout})
            
            
            t_result.append([l_t1,l_v1,l_test1,l_t2,l_v2,l_test2,l_t3,l_v3,l_test3])
        
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))
        csv_pd = pd.DataFrame(t_result)  
        csv_pd.to_excel('./training_result.xlsx', sheet_name='training',header=False,index = False)
        
        saver = tf.compat.v1.train.Saver()
        #saver_path = saver.save(self.sess, "./Model/model.ckpt")
        #print ("Model saved in file: ", saver_path)
    
        
    def test_model1(self,t_data,outmax,outmin,num,name):
        #old0
        #model1
        output0=[]
        targets=[]
        for i in range(self.lag,t_data.shape[0]-1):
            batch_xs=np.array([t_data[i,:]])
            batch_ys=np.array([t_data[i+1,:]])
            batch_xs.reshape([1,self.in_size])
            pre=self.sess.run(self.out_1,feed_dict={self.x_1:batch_xs})      
            result=((pre+1)/2)*(0.001+outmax-outmin)+outmin
            target=((batch_ys+1)/2)*(0.001+outmax-outmin)+outmin
            output0.append([result]) 
            targets.append(target)
        result0=np.array([output0])
        true=np.array([targets])
        result0=result0.reshape(t_data.shape[0]-self.lag-1,t_data.shape[1])
        true=true.reshape(t_data.shape[0]-self.lag-1,t_data.shape[1])
        return result0,true
            
    def test_model2(self,t_data,outmax,outmin,training_datain,training_dataout,num,name):
        #old1
        #model2
        #add data training
        for i in range(10):
            self.sess.run(self.train2,feed_dict={self.x_2:training_datain,self.y_2:training_dataout})
        output0=[]
        for i in range(self.lag,t_data.shape[0]-1):
            batch_xs=np.array([t_data[i,:]])
            batch_xs.reshape([1,self.in_size])
            pre=self.sess.run(self.out_2,feed_dict={self.x_2:batch_xs})      
            result=((pre+1)/2)*(0.001+outmax-outmin)+outmin
            output0.append([result])
            
        result0=np.array([output0])
        result0=result0.reshape(t_data.shape[0]-self.lag-1,t_data.shape[1])
        return result0

    def test_model3(self,t_data,outmax,outmin,training_datain,training_dataout,num,name):
        #new
        #model3
        #add data training
        for i in range(50):
            self.sess.run(self.train2,feed_dict={self.x_2:training_datain,self.y_2:training_dataout})
        output0=[]
        for i in range(self.lag,t_data.shape[0]-1):
            #online training
            t_temx=t_data[i-1-6:i-1,:]
            t_temy=t_data[i-1-6+1:i,:]
            bx=np.array(t_temx)
            by=np.array(t_temy)
            for t in range(20):  
                self.sess.run(self.train3,feed_dict={self.x_3:bx,self.y_3:by})
            
            batch_xs=np.array([t_data[i,:]])
            batch_xs.reshape([1,self.in_size])
            pre=self.sess.run(self.out_3,feed_dict={self.x_3:batch_xs})      
            result=((pre+1)/2)*(0.001+outmax-outmin)+outmin
            output0.append([result]) 
            
        result0=np.array([output0])
        result0=result0.reshape(t_data.shape[0]-self.lag-1,t_data.shape[1])
        return result0
            
    def loc_test(self):
        name='loc'
        saver = tf.compat.v1.train.Saver()
        self.sess=tf.compat.v1.Session()
        saver.restore(self.sess, "./Model/model.ckpt")
        
        writer = pd.ExcelWriter('./loc_results.xlsx')

        for num in range(3):
            if num==0:
                [tnum,xnum]=self.t_data1.shape
                t_data=self.t_data1[int(0.25*tnum):int(0.5*tnum),:]
                
                training_datain=self.t_data1[0:int(0.25*tnum),:]
                training_dataout=self.t_data1[1:int(0.25*tnum)+1,:]
                
                outmax=self.outmax1
                outmin=self.outmin1
            elif num==1:
                [tnum,xnum]=self.t_data2.shape
                t_data=self.t_data2[int(0.25*tnum):int(0.5*tnum),:]
                
                training_datain=self.t_data2[0:int(0.25*tnum),:]
                training_dataout=self.t_data2[1:int(0.25*tnum)+1,:]
                
                outmax=self.outmax2
                outmin=self.outmin2
            else:
                [tnum,xnum]=self.t_data3.shape
                t_data=self.t_data3[int(0.25*tnum):int(0.5*tnum),:]
                
                training_datain=self.t_data3[0:int(0.25*tnum),:]
                training_dataout=self.t_data3[1:int(0.25*tnum)+1,:]
                
                outmax=self.outmax3
                outmin=self.outmin3
            
            old0,true=self.test_model1(t_data,outmax,outmin,num,name)
            old1=self.test_model2(t_data,outmax,outmin,training_datain,training_dataout,num,name)
            new=self.test_model3(t_data,outmax,outmin,training_datain,training_dataout,num,name)
            
            result=np.concatenate((new,old0,old1,true),axis=1)
            csv_pd = pd.DataFrame(result)  
            csv_pd.to_excel(writer, sheet_name='data'+str(num),header=False,index = False)
            
            
    def time_test(self):
        name='time'
        saver = tf.compat.v1.train.Saver()
        self.sess=tf.compat.v1.Session()
        saver.restore(self.sess, "./Model/model.ckpt")
        
        a=0.3
        
        [tnum,xnum]=self.t_data0.shape
        t_data=self.t_data0[int(a*tnum):tnum,:]
        training_datain=self.t_data0[0:int(a*tnum),:]
        training_dataout=self.t_data0[1:int(a*tnum)+1,:]       
        outmax=self.outmax0
        outmin=self.outmin0
        num=4
        old0,true=self.test_model1(t_data,outmax,outmin,num,name)
        old1=self.test_model2(t_data,outmax,outmin,training_datain,training_dataout,num,name)
        new=self.test_model3(t_data,outmax,outmin,training_datain,training_dataout,num,name)
        
        result=np.concatenate((new,old0,old1,true),axis=1)
        
        writer = pd.ExcelWriter('./time_results.xlsx')
        csv_pd = pd.DataFrame(result)  
        csv_pd.to_excel(writer, sheet_name='data'+str(num),header=False,index = False)
        
    
    def mix_test(self):
        name='mix'
        saver = tf.compat.v1.train.Saver()
        self.sess=tf.compat.v1.Session()
        saver.restore(self.sess, "./Model/model.ckpt")
        
        writer = pd.ExcelWriter('./mix_results.xlsx')
        
        for num in range(3):
            if num==0:
                [tnum,xnum]=self.t_data1.shape
                t_data=self.t_data1[int(0.25*tnum):tnum,:]
                training_datain=self.t_data1[0:int(0.25*tnum),:]
                training_dataout=self.t_data1[1:int(0.25*tnum)+1,:]
                outmax=self.outmax1
                outmin=self.outmin1
            elif num==1:
                [tnum,xnum]=self.t_data2.shape
                t_data=self.t_data2[int(0.25*tnum):tnum,:]
                training_datain=self.t_data2[0:int(0.25*tnum),:]
                training_dataout=self.t_data2[1:int(0.25*tnum)+1,:]
                outmax=self.outmax2
                outmin=self.outmin2
            else:
                #print('runnning',num)
                [tnum,xnum]=self.t_data3.shape
                t_data=self.t_data3[int(0.25*tnum):tnum,:]
                training_datain=self.t_data3[0:int(0.25*tnum),:]
                training_dataout=self.t_data3[1:int(0.25*tnum)+1,:]
                outmax=self.outmax3
                outmin=self.outmin3
            
            old0,true=self.test_model1(t_data,outmax,outmin,num,name)
            old1=self.test_model2(t_data,outmax,outmin,training_datain,training_dataout,num,name)
            new=self.test_model3(t_data,outmax,outmin,training_datain,training_dataout,num,name)
            
            result=np.concatenate((new,old0,old1,true),axis=1)
            csv_pd = pd.DataFrame(result)  
            csv_pd.to_excel(writer, sheet_name='data'+str(num),header=False,index = False)

if __name__=='__main__':
    wq_p=wq_pred()
    wq_p._train()
    #wq_p.loc_test()
    #wq_p.time_test()
    #wq_p.mix_test()
           