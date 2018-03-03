
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[2]:

dis_calculation = 0
def euclidean(a,b):
    d = np.linalg.norm(a-b)
    global dis_calculation 
    dis_calculation = dis_calculation + d
    return d


# In[3]:

import math
def euclidean2(a,b):
    distance = 0 
    for i in range(len(a)):
        distance += (a[i]-b[i])*(a[i]-b[i])

#    print 'np',np.sqrt(distance)
    return np.sqrt(distance)


# In[4]:

def create_centers(k,dataset):
    init_centers = {}
    for i in range(k):
        init_centers[i] = dataset[i]
    centers = init_centers
    return centers


# In[5]:

def create_centers2(k,dataset):
    init_centers = {}
    for i in range(k):
        init_centers[i] = dataset[int(np.random.choice(len(dataset), size = 1, replace = False))]
    centers = init_centers
    return centers


# In[6]:

def compute_s(k,dataset,centers):
    sj = {}
    
    for j in range(k):
        min_d = []
        for j2 in range(k):
            if j != j2:
                d = euclidean(np.asarray(centers[j]),np.asarray(centers[j2]))/2
                min_d.append(d)
        sj[j] = min(min_d)
    return sj


# In[7]:

def check_converge(max_iter, treshold,move_d,iter_times):
    while iter_times < max_iter:
        d = 0.0
        for i in range(len(move_d)):
            d += move_d[i]
        #print 'd',d
        if d < treshold:
            return True
        else:
            return False


# In[8]:

def sse(centers,clustering):
    sse = 0 
    for i in clustering:
        for data in clustering[i]:
            d = np.linalg.norm(np.asarray(data) - centers[i])
            
            sse += d**2
    return sse


# In[9]:

def regenerate_centers(k,dataset,a,func,max_iter,treshold):
    new_centers = {}
    clustering = {}
    for j in range(k):
        clustering[j] = []
    for j in range(k):    
        for i in range(len(a)):
            if a[i] == j:
                clustering[j].append(dataset[i])
    
#    for j in clustering:
#            if len(clustering[j]) == 0:
#                elkan(k,dataset,func, max_iter,treshold )
#    print 'clustering',clustering
    for j in clustering:
        new_centers[j] = np.average(clustering[j],axis = 0)
    return new_centers, clustering    


# In[10]:

def elkan(k,dataset,func, max_iter,treshold ):
    centers = create_centers(k,dataset)
    print 'init_center',centers
    length = len(dataset)
    a = np.full(length, 1)
    u = np.full(length, float(999))
    l = np.full((length,k),float(0))
 #   print a[:5], u[:5],l[:5]
    iter_times = 0
    check_conver = False
    while check_conver == False:
        sj = compute_s(k,dataset,centers)
      
        for i in range(len(dataset)):
            if u[i] <= sj[a[i]]:
                continue
            r = True
            for j in range(k):
 #               print 'ij', i ,j
                z = []
                z.append(l[i,j])
                z.append(func(np.asarray(centers[a[i]]),np.asarray(centers[j]))/2)
                z = max(z)
 #               print 'z', z
                if j == a[i] or u[i] <= z:
                    continue
                if r:
 #                   print 'data',dataset[i]
 #                   print 'data centers',centers[a[i]]
                    u[i] = float(func(np.asarray(dataset[i]), np.asarray(centers[a[i]])))
 #                   print 'u', u[i]
                    r = False
                    if u[i] <= z:
                        continue                
                l[i,j] = func(np.asarray(dataset[i]),np.asarray(centers[j]))
#                print 'l',l[i,j]
                if l[i,j] < u[i]:
                    a[i] = j
                    r = True
        new_centers, clustering = regenerate_centers(k,dataset,a,func,max_iter,treshold)
        #print '******'
        #print 'new',new_centers
        move_d = {}
        
        for j in range(k):
            #print 'center', centers[j]
            move_d[j] = func(new_centers[j],centers[j])
        #print move_d
        for i in range(len(dataset)):
            u[i] = u[i] + move_d[a[i]]
            for j in range(k):
                if (l[i,j] - move_d[j]) >= 0:
                    l[i,j] = l[i,j] - move_d[j] 
                else:
                    l[i,j] = 0
        
        check_conver = check_converge(max_iter, treshold,move_d,iter_times)
        if check_conver == False:
            iter_times += 1
            centers = new_centers.copy() 
    global dis_calculation 
    sum_of_square = sse(new_centers,clustering)   
    print 'number of iterations:', iter_times
    print 'distance calculations:', int(dis_calculation)
    print 'sum of suare error:', int(sum_of_square)
    return new_centers, clustering
            


# In[11]:

def turn_list(df):
    ls = []
    for index, row in df.iterrows():
        ls.append(list(row))
    return ls


# In[12]:

def predict(centers, data, func):
    distance = []
    for center in centers:
        d = func(np.asarray(data) , centers[center])
        distance.append(d)
    center_index = distance.index(min(distance))
 
    return center_index


# In[13]:

def evaluate(dataset,k,func, max_iter, treshold):
    label = []
    datasetX = []
    for data in dataset:
        label.append(data[-1])
        datasetX.append(data[:-1])
    
    elkanV = elkan(k,datasetX,func, max_iter,treshold)
    centers = elkanV[0]
    predictions = []
    for data in datasetX:        
        prediction = predict(centers,data,func)
        predictions.append(prediction)
        
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == label[i]:
            count +=1

    print 'accuacy:', float(count)/float(len(predictions))*100,'%' 


# In[14]:

k = 3
max_iter = 100
treshold = 0.0001

df = pd.read_csv('/Users/eddiewu/downloads/kcupcut.csv')
normalized_df=(df-df.mean())/df.std()
#sub_data = normalized_df.iloc[:,0:5]
dataset = turn_list(normalized_df)
elkan(k,dataset,euclidean,max_iter, treshold)


# In[21]:

k = 20
max_iter = 100
treshold = 0.0001
elkan(k,dataset,euclidean,max_iter, treshold)


# In[48]:

k = 100
max_iter = 100
treshold = 0.0001
elkan(k,dataset,euclidean,max_iter, treshold)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



