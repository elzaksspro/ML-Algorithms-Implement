
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

def create_centers(k,dataset):
    init_centers = {}
    for i in range(k):
        init_centers[i] = dataset[i]

    centers = init_centers
    
    return centers


# In[3]:

def create_centers2(k,dataset):
    init_centers = {}
    for i in range(k):
        init_centers[i] = dataset[int(np.random.choice(len(dataset), size = 1, replace = False))]
    centers = init_centers
    return centers


# In[4]:

from scipy import spatial
dis_calculation = 0
def euclidean(a,b):
    d = np.linalg.norm(a-b)
    global dis_calculation 
    dis_calculation = dis_calculation + d
    return d


def cosine(a,b):
    return spatial.distance.cosine(a, b)

def sumofsquare(x):
    sum = 0
    for i in x:
        sum += i**2
    return sum    

def cosine2(a,b):
    numerator = 0
    denumerator = 0
    for i in range(len(a)):
        numerator += a[i]*b[i]
        
    denumerator += math.sqrt(sumofsquare(a))*math.sqrt(sumofsquare(b))     
    return 1-(float(numerator)/float(denumerator))

def cityblock(a,b):
    return spatial.distance.cityblock(a, b)

def citiblockD(a,b):
    distance = 0 
    for i in range(len(a)):
        distance += abs(a[i]-b[i])
    return distance

def equ1(a,b):
    distance1 = 0
    distance2 = 0
    for i in range(len(a)):

        if a[i]>b[i]:
            distance1 += a[i]-b[i]
        elif b[i]>a[i]:
            distance2 += b[i]-a[i]
    return (distance1**2 + distance2**2)**(0.5)


# In[5]:

def equ2(a,b):
    distance1 = 0
    distance2 = 0
    sigma = 0
    for i in range(len(a)):
        if a[i]>b[i]:
            distance1 += a[i]-b[i]
        elif b[i]>a[i]:
            distance2 += b[i]-a[i]
        find_max = []
        find_max.append(abs(a[i]))
        find_max.append(abs(b[i]))
        find_max.append(abs(a[i]-b[i]))
        found = max(find_max)
        sigma += found
        
    return (distance1**2 + distance2**2)**(0.5)/sigma
    

    


# In[6]:

def regenerate_centers(clustering):
    new_centers = {}
    
    for center in clustering:
        new_centers[center] = np.average(clustering[center],axis = 0)
    return new_centers    


# In[7]:

#def visualize(centers,clustering):
#    colors = 10*["r", "g", "c", "b", "k"]
#    for center in centers:
##        plt.scatter(centers[center][0], centers[center][1], s = 130, marker = "x")
 #   for i in clustering:
 #       color = colors[i]
 #       for data in clustering[i]:
 ##           plt.scatter(data[0], data[1], color = color,s = 30)
 #   plt.show()


    


# In[8]:

def sse(clustering,centers):
    sse = 0 
    for i in clustering:
        for data in clustering[i]:
            d = np.linalg.norm(np.asarray(data) - centers[i])
            
            sse += d**2
    return sse
        


# In[9]:


def clf(dataset, centers,k, distance_func):
    
    clustering = {}
    for i in range(k):
        clustering[i] = []
    
    
    for data in dataset:
        distance = []
        
        for center in centers:
            d = distance_func(np.asarray(data) , np.asarray(centers[center]))
#            d = np.linalg.norm(np.asarray(data) - np.asarray(centers[center]))
            distance.append(d)
        cluster = distance.index(min(distance))
        clustering[cluster].append(data)
    
    return clustering


# In[10]:

def kmean_clustering(k,dataset,distance_func,max_iter, treshold):
    centers = create_centers(k,dataset)
    print 'init_centers',centers
    clustering = clf(dataset, centers,k,distance_func)
    
    
    iter_time = 0
    while iter_time < max_iter:
        new_centers = regenerate_centers(clustering)
        #print 'new', new_centers

        
        new_clustering = clf(dataset, new_centers,k,distance_func)
#        print 'new center value',new_centers.values()
#        print 'list', list(new_centers.values())
#        print 'cluster',np.asarray(list(new_centers.values()))
      #  d = distance_func(np.asarray(list(new_centers.values())) , np.asarray(list(centers.values())))
        move = 0
        for i in range(k):
            d = distance_func(np.asarray(new_centers.values()[i]) , np.asarray(centers.values()[i]))
            move += d
        if move < treshold:
            break
        else:            
            centers = new_centers
                #print 'centerX', centers
            clustering = new_clustering
            #print 'clusteringX', clustering
            iter_time += 1
            #print iter_time
    global dis_calculation     
    sum_of_square = sse(new_clustering, new_centers)   
    print 'number of iterations:', iter_time
    print 'distance calculations:', int(dis_calculation)
    print 'sum of suare error:', int(sum_of_square)

    return  new_centers,new_clustering


# In[11]:

def predict(centers, data, distance_func):
    distance = []
#    print 'cccc',centers
    for center in centers:

        d = distance_func(np.asarray(data) , centers[center])
        distance.append(d)
    center_index = distance.index(min(distance))
 
    return center_index


# In[13]:

def kmean(dataset,k,distance_func,max_iter, treshold):    
    kmean = kmean_clustering(k,dataset,distance_func,max_iter, treshold)
    predictions = []

    new_centers = kmean[0] 
    for data in dataset:
        
        prediction = predict(new_centers,data,distance_func)
        predictions.append(prediction)
    return predictions
    


# In[14]:

def evaluate(dataset,k,distance_func, max_iter, treshold):
    label = []
    datasetX = []

    for data in dataset:
        label.append(data[-1])
        datasetX.append(data[:-1])
#    print 'label',label            
    predictions = kmean(datasetX,k,distance_func,max_iter, treshold)
#    print 'pred',predictions
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == label[i]:
            count +=1

    print 'accuacy:', float(count)/float(len(predictions))*100,'%' 


# In[15]:

def turn_list(df):
    ls = []
    for index, row in df.iterrows():
        ls.append(list(row))
    return ls


# # Question 3 part a 
# retrun number of interations, distance calculations and total sum of squared

# In[16]:

data = pd.read_csv('data.csv',header =None)
k = 3
max_iter = 100
treshold = 0.0001
kmean_clustering(k,turn_list(data),euclidean,max_iter, treshold)


# # Question 3 part b
# Compare basic kmean performance with elkan kmean 

# In[17]:

k = 3
max_iter = 100
treshold = 0.0001

df = pd.read_csv('/Users/eddiewu/downloads/kcupcut.csv')
normalized_df=(df-df.mean())/df.std()
#sub_data = normalized_df.iloc[:,0:5]
dataset = turn_list(normalized_df)
kmean_clustering(k,dataset,euclidean,max_iter, treshold)


# In[146]:

k = 20
max_iter = 100
treshold = 0.0001
kmean_clustering(k,dataset,euclidean,max_iter, treshold)


# In[151]:

k = 100
max_iter = 300
treshold = 0.0001

df = pd.read_csv('/Users/eddiewu/downloads/kcupcut.csv')
normalized_df=(df-df.mean())/df.std()
#sub_data = normalized_df.iloc[:,0:5]
dataset = turn_list(normalized_df)
kmean_clustering(k,dataset,euclidean,max_iter, treshold)


# # Question 3 part C
# implement various distance function

# In[18]:

k = 3
max_iter = 100
treshold = 0.000001

df = pd.read_csv('/Users/eddiewu/downloads/bezdekIris.data.csv', header = None)
df.loc[df[4] == 'Iris-setosa',4] = 2
df.loc[df[4] == 'Iris-versicolor',4] = 1
df.loc[df[4] == 'Iris-virginica',4] = 0
dataset = turn_list(df)
print 'euclidean'
evaluate(dataset,3,euclidean,max_iter, treshold)
print '*****'
print 'equation 1'
evaluate(dataset,3,equ1,max_iter, treshold)
print '*****'
print 'equation 2'
evaluate(dataset,3,equ2,max_iter, treshold)
print '*****'
print 'cityblock'
evaluate(dataset,3,cityblock,max_iter, treshold)
print '*****'
print 'cosine'
evaluate(dataset,3,cosine,max_iter, treshold)


# # Question3 Part D
# Apply k-mean on tree dataset

# In[19]:

#Iris data Set
k = 3
max_iter = 100
treshold = 0.000001

df = pd.read_csv('/Users/eddiewu/downloads/bezdekIris.data.csv', header = None)
df.loc[df[4] == 'Iris-setosa',4] = 2
df.loc[df[4] == 'Iris-versicolor',4] = 1
df.loc[df[4] == 'Iris-virginica',4] = 0
dataset = turn_list(df)
evaluate(dataset,3,euclidean,max_iter, treshold)


# In[20]:

df = pd.read_csv('/Users/eddiewu/downloads/wine.csv', header = None)
df.head()


# In[21]:

#Wine data Set
k = 3
max_iter = 100
treshold = 0.0001

df = pd.read_csv('/Users/eddiewu/downloads/wine.csv', header = None)
#df[0] = df[0].astype('category')
temp = df.iloc[:,1:]
normalized_df=(temp-temp.mean())/temp.std()
normalized_df[14] = df[0].copy()
normalized_df.loc[normalized_df[14] == 1, 14] = 0
normalized_df.loc[normalized_df[14] == 2, 14] = 1
normalized_df.loc[normalized_df[14] == 3, 14] = 2
#del df[0]
#df = df.drop(df.loc[:,0])
dataset = turn_list(normalized_df)
evaluate(dataset,3,euclidean,max_iter, treshold)


# In[22]:

#balance data Set
k = 3
max_iter = 100
treshold = 0.0001

df = pd.read_csv('/Users/eddiewu/downloads/balance-scale.csv', header = None)
#df[0] = df[0].astype('category')
temp = df.iloc[:,1:]

df[5] = df[0].copy()
df.loc[df[5] == 'R', 5] = 2
df.loc[df[5] == 'L', 5] = 1
df.loc[df[5] == 'B', 5] = 0
del df[0]

dataset = turn_list(df)
evaluate(dataset,3,euclidean,max_iter, treshold)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



