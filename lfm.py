# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import random
from numpy.linalg import norm
import os
from reviews_parser import generate_extended_reviews_file


file_path_separator = '/' if os.name == 'posix' else '\\'
base_dir = 'test_dataset' + file_path_separator

############## Preprocessing
generate_extended_reviews_file()
reviews = pd.read_csv(base_dir + 'reviews_extended.csv')

rows=reviews.reviewers.unique()
columns=reviews.course_id.unique()

rows_ln = len(rows)
columns_ln = len(columns)

myData = np.array([0.0 for i in range(rows_ln*columns_ln)])
mydf = pd.DataFrame(myData.reshape(rows_ln, -1))

mydf.columns=columns
mydf.index=rows

print(mydf.head())

print('Filling DataFrame, please wait...')
reviews_ln = len(reviews)
for i in range(reviews_ln):
    mydf.loc[reviews.loc[i,'reviewers'],reviews.loc[i,'course_id']] = reviews.loc[i,'vader_score']

print(mydf.head())

reviewers_list=list(mydf.index)
courses_list=list(mydf.columns)
R = coo_matrix(mydf.values)

print ("R Shape::", R.shape)
print ("R Columns::", R.col)
print ("R Rows::",R.row)

M,N=R.shape
# No of Factors - 3
K=3
# using random values of P and Q 
P=np.random.rand(M,K)
Q=np.random.rand(K,N)

############## Splitting the Dataset into Training Set and Testing Set
r_data_len = len(R.data)
train_index=random.sample([i for i in range(r_data_len)],int(r_data_len * 0.8))

test_index = set([i for i in range(r_data_len)])-set(train_index)

test_index = list(test_index)

########## Using Gradient Descending Method to Approximate P, Q

def error(R,P,Q,index,lamda=0.02):
    ratings = R.data[index]
    rows = R.row[index]
    cols = R.col[index]
    e = 0 
    for ui in range(len(ratings)):
        rui=ratings[ui]
        u = rows[ui]
        i = cols[ui]
        if rui>0:
            e= e + pow(rui-np.dot(P[u,:],Q[:,i]),2)+\
                lamda*(pow(norm(P[u,:]),2)+pow(norm(Q[:,i]),2))
    return e

error(R,P,Q,train_index)

error(R,P,Q,test_index)

def SGD(R,K,train_index,test_index,lamda=0.02,steps=10,gamma=0.001):
    
    M,N = R.shape
    P = np.random.rand(M,K)
    Q = np.random.rand(K,N)
    
    rmse = np.sqrt(error(R,P,Q,test_index,lamda)/len(R.data[test_index]))
    print("Initial RMSE: "+str(rmse))
    
    for step in range(steps):
        for ui in range(len(R.data[train_index])):
            rui=R.data[ui]
            u = R.row[ui]
            i = R.col[ui]
            if rui>0:
                eui=rui-np.dot(P[u,:],Q[:,i])
                P[u,:]=P[u,:]+gamma*2*(eui*Q[:,i]-lamda*P[u,:])
                Q[:,i]=Q[:,i]+gamma*2*(eui*P[u,:]-lamda*Q[:,i])
        rmse = np.sqrt(error(R,P,Q,test_index,lamda)/len(R.data[test_index]))
        if rmse<0.5:
            break
    print("Final RMSE: "+str(rmse))
    return P,Q,round(rmse, 4)

value = []
for k in range(10,20):
    print('k=',k)
    P,Q,rmse=SGD(R,K=k,train_index=train_index,test_index=test_index,gamma=0.0007,lamda=0.01, steps=100)
    value.append(rmse)
    print('------------------------')

plt.plot(range(10,20),value)
plt.xlabel("Number of latent factors")
plt.ylabel("RMSE on test set")
plt.show()

# P,Q=SGD(R,K=14,train_index=train_index,test_index=test_index,gamma=0.0007,lamda=0.01, steps=100)

# np.sqrt(error(R,P,Q,test_index,0.01)/len(R.data[test_index]))

# ## np.matmul(P, Q)

# est_df = pd.DataFrame(np.round(np.matmul(P,Q),4),columns=courses_list, index=reviewers_list)