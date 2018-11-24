# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:41:34 2018

@author: chong
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:50:26 2018

@author: chong

The ANN model with trans and online-training 
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
outmax=max_data(training_data)
outmin=min_data(training_data)
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
h1_size=8
h2_size=8
h3_size=4
h4_size=4
out_size=1
lr=0.001
steps=100
batch_size=100

x_=tf.placeholder(tf.float32,[None,in_size])
y_=tf.placeholder(tf.float32,[None,out_size])

W1=tf.Variable(tf.truncated_normal([in_size,h1_size],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_size]))
W2=tf.Variable(tf.truncated_normal([h1_size,h2_size],stddev=0.1))
b2=tf.Variable(tf.zeros([h2_size]))
W3=tf.Variable(tf.truncated_normal([h2_size,h3_size],stddev=0.1))
b3=tf.Variable(tf.zeros([h3_size]))
W4=tf.Variable(tf.truncated_normal([h3_size,h4_size],stddev=0.1))
b4=tf.Variable(tf.zeros([h4_size]))
W5=tf.Variable(tf.truncated_normal([h4_size,out_size],stddev=0.1))
b5=tf.Variable(tf.zeros([out_size]))

h1=tf.nn.tanh(tf.matmul(x_,W1)+b1)
h2=tf.nn.tanh(tf.matmul(h1,W2)+b2)
h3=tf.nn.tanh(tf.matmul(h2,W3)+b3)
h4=tf.nn.tanh(tf.matmul(h3,W4)+b4)
out_=tf.nn.tanh(tf.matmul(h4,W5)+b5)

loss=tf.sqrt(tf.reduce_mean(tf.square(out_ - y_)))
train=tf.train.AdamOptimizer(lr).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(steps):
        for i in range(training_datain.shape[0]-batch_size-1):
            bx=training_datain[i:i+batch_size,:]
            bx.reshape([batch_size,in_size])
            by=training_dataout[i:i+batch_size,:]
            by.reshape([batch_size,out_size])
            sess.run(train,feed_dict={x_:bx,y_:by})

    #测试1
    output1=[]
    for i in range(test_datain1.shape[0]-1):
        batch_xs=np.array(test_datain1[i:i+1,:])
        batch_xs.reshape([1,7])
        pre=sess.run(out_,feed_dict={x_:batch_xs})      
        result=((pre+1)/2)*(outmax1[-1]-outmin1[-1])+outmin1[-1]
        output1.append([result])
        bx=test_datain1[i:i+1,:]
        bx.reshape([1,in_size])
        by=test_datain1[i+1:i+2,test_datain1.shape[1]-1:test_datain1.shape[1]]
        by.reshape([1,out_size])
        sess.run(train,feed_dict={x_:bx,y_:by})
    result1=np.array(output1)
    result1=result1.reshape(test_datain1.shape[0]-1,1)
    #测试2
    output2=[]
    for i in range(test_datain2.shape[0]-1):
        batch_xs=np.array(test_datain2[i:i+1,:])
        batch_xs.reshape([1,7])
        pre=sess.run(out_,feed_dict={x_:batch_xs})      
        result=((pre+1)/2)*(outmax2[-1]-outmin2[-1])+outmin2[-1]
        output2.append([result])
        bx=test_datain2[i:i+1,:]
        bx.reshape([1,in_size])
        by=test_datain2[i+1:i+2,test_datain2.shape[1]-1:test_datain2.shape[1]]
        by.reshape([1,out_size])
        sess.run(train,feed_dict={x_:bx,y_:by})
    result2=np.array(output2)
    result2=result2.reshape(test_datain2.shape[0]-1,1)     
    #测试3
    output3=[]
    for i in range(test_datain3.shape[0]-1):
        batch_xs=np.array(test_datain3[i:i+1,:])
        batch_xs.reshape([1,7])
        pre=sess.run(out_,feed_dict={x_:batch_xs})      
        result=((pre+1)/2)*(outmax3[-1]-outmin3[-1])+outmin3[-1]
        output3.append([result])
        bx=test_datain3[i:i+1,:]
        bx.reshape([1,in_size])
        by=test_datain3[i+1:i+2,test_datain3.shape[1]-1:test_datain3.shape[1]]
        by.reshape([1,out_size])
        sess.run(train,feed_dict={x_:bx,y_:by})
    result3=np.array(output3)
    result3=result3.reshape(test_datain3.shape[0]-1,1)   

    np.savetxt("ANN_test1_result.txt", result1,delimiter="\n")
    np.savetxt("ANN_test2_result.txt", result2,delimiter="\n")
    np.savetxt("ANN_test3_result.txt", result3,delimiter="\n")
    #save_data('ANN_test1_result',result1)
    #save_data('ANN_test2_result',result2) 
    #save_data('ANN_test3_result',result3)      