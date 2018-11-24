# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:50:26 2018

@author: chong

The RNN model with online training
"""


import tensorflow as tf
import numpy as np
import xlrd
import xlwt

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
            data1[j][i]=2*(data1[j][i]-amin[i])/(amax[i]-amin[i])-1
        
    return data1


def read_data(st):
    tr_data=[]
    data=xlrd.open_workbook(st)
    table=data.sheets()[0]
    nrows=table.nrows
    ncols=table.ncols
    for i in range(nrows):
        tem=[]
        for j in range(ncols):
            tem.append(table.cell(i,j).value)     
        tr_data.append(tem)
    t_data=np.array(tr_data)
    t_data.reshape(nrows,ncols)
    print(t_data.shape)
    return t_data

def save_data(st,data):
    wb=xlwt.Workbook()
    ws=wb.add_sheet(st)
    for i in range(data.shape[0]):
        ws.write(i,1,data[i][0])
        
    wb.save(st+'.xls')

#训练数据
training_data=read_data('training_data.xls')
#print(training_data[0:training_data.shape[0]-1,:])
#print(training_data[1:training_data.shape[0],6:7])
#测试数据1
test1_data=read_data('test_data.xls')
#测试数据2
test2_data=read_data('test_data2.xls')
#测试数据3
test3_data=read_data('test_data3.xls')

#归一化
#outmax=max_data(training_data)
#outmin=min_data(training_data)
outmax1=max_data(test1_data)
outmin1=min_data(test1_data)
outmax2=max_data(test2_data)
outmin2=min_data(test2_data)
outmax3=max_data(test3_data)
outmin3=min_data(test3_data)

test_datain1=normalization(test1_data)[:,:]
test_datain2=normalization(test2_data)[:,:]
test_datain3=normalization(test3_data)[:,:]

t_data=normalization(training_data)
training_datain=t_data[0:t_data.shape[0]-1,:]
training_dataout=t_data[1:t_data.shape[0],6:7]

#print(training_datain.shape)
#print(training_dataout.shape)

#计算图
in_size=7
step_size=3
h_size=8
out_size=1
lr=0.001
steps=1000
batch_size=1

x_=tf.placeholder(tf.float32,[None,step_size,in_size])
y_=tf.placeholder(tf.float32,[None,out_size])

weights={
'in':tf.Variable(tf.random_normal([in_size,h_size])),
'out':tf.Variable(tf.random_normal([h_size,out_size]))}

biases={
'in':tf.Variable(tf.random_normal([h_size,])),
'out':tf.Variable(tf.random_normal([out_size,]))}

X=tf.reshape(x_,[-1,in_size])   
X_in=tf.matmul(X,weights['in'])+biases['in']
X_in=tf.nn.tanh(tf.reshape(X_in,[-1,step_size,h_size]))#(128batch 28steps 128hidden) 
#cell
lstm_cell0=tf.nn.rnn_cell.BasicLSTMCell(h_size,forget_bias=0.8) #For LSTM
#lstm_cell0=tf.nn.rnn_cell.BasicRNNCell(h_size) #For RNN
cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell0)
lstm_cell = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(3)])
#output,_=tf.nn.rnn(lstm_cell,X_in,dtype=tf.float32)
_init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
outputs,states=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False) 
#out_=tf.nn.tanh(tf.matmul(states[-1],weights['out'])+biases['out']) #For RNN
out_=tf.nn.tanh(tf.matmul(states[-1][-1],weights['out'])+biases['out']) #For LSTM

loss=tf.sqrt(tf.reduce_mean(tf.square(out_ - y_)))
train=tf.train.AdamOptimizer(lr).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(steps):
        for i in range(training_datain.shape[0]-batch_size-step_size-1):
            temx=[]
            temy=[]
            for j in range(batch_size):
                temx.append(training_datain[i+j:i+j+step_size,:])
                temy.append(training_dataout[i+j+step_size-1])
            bx=np.array(temx)
            bx.reshape([batch_size,step_size,in_size])
            by=np.array(temy)
            by.reshape([batch_size,out_size])
            sess.run(train,feed_dict={x_:bx,y_:by})


    #测试1
    output1=[]
    for i in range(test_datain1.shape[0]-batch_size-step_size-1):
        temx=[]
        temy=[]
        for j in range(batch_size):
            temx.append(test_datain1[i+j:i+j+step_size,:])
            temy.append([test_datain1[i+j+step_size-1,in_size-1]])
        bx=np.array(temx)
        bx.reshape([batch_size,step_size,in_size])
        by=np.array(temy)
        by.reshape([batch_size,out_size])
        pre=sess.run(out_,feed_dict={x_:bx})      
        result=((pre+1)/2)*(outmax1[-1]-outmin1[-1])+outmin1[-1]
        output1.append([result])
        sess.run(train,feed_dict={x_:bx,y_:by})
    result1=np.array(output1)
    result1=result1.reshape(test_datain1.shape[0]-batch_size-step_size-1,1)
    
    #测试2
    output2=[]
    for i in range(test_datain2.shape[0]-batch_size-step_size-1):
        temx=[]
        temy=[]
        for j in range(batch_size):
            temx.append(test_datain2[i+j:i+j+step_size,:])
            temy.append([test_datain2[i+j+step_size-1,in_size-1]])
        batch_xs=np.array(temx)
        batch_xs.reshape([batch_size,step_size,in_size])
        batch_ys=np.array(temy)
        batch_ys.reshape([batch_size,out_size])
        pre=sess.run(out_,feed_dict={x_:batch_xs})      
        result=((pre+1)/2)*(outmax2[-1]-outmin2[-1])+outmin2[-1]
        output2.append([result])
        sess.run(train,feed_dict={x_:batch_xs,y_:batch_ys})
    result2=np.array(output2)
    result2=result2.reshape(test_datain2.shape[0]-batch_size-step_size-1,1)
    
    #测试3
    output3=[]
    for i in range(test_datain3.shape[0]-batch_size-step_size-1):
        temx=[]
        temy=[]
        for j in range(batch_size):
            temx.append(test_datain3[i+j:i+j+step_size,:])
            temy.append([test_datain3[i+j+step_size-1,in_size-1]])
        batch_xs=np.array(temx)
        batch_xs.reshape([batch_size,step_size,in_size])
        batch_ys=np.array(temy)
        batch_ys.reshape([batch_size,out_size])
        pre=sess.run(out_,feed_dict={x_:batch_xs})      
        result=((pre+1)/2)*(outmax3[-1]-outmin3[-1])+outmin3[-1]
        output3.append([result])
        sess.run(train,feed_dict={x_:batch_xs,y_:batch_ys})
    result3=np.array(output3)
    result3=result3.reshape(test_datain3.shape[0]-batch_size-step_size-1,1)


    np.savetxt("RNN_test1_result1000.txt", result1,delimiter="\n")
    np.savetxt("RNN_test2_result1000.txt", result2,delimiter="\n")
    np.savetxt("RNN_test3_result1000.txt", result3,delimiter="\n")
    
    #save_data('RNN_test1_result',result1)
    #save_data('ANN_test2_result',result2) 
    #save_data('ANN_test3_result',result3)      
