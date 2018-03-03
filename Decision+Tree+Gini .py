
# coding: utf-8

# In[172]:

def gini_calculator(groups,classes):
    #we are calculating the gini number in each group and sum it up    
    N = 0
    gini = 0
    #caculate the total number of data
    for group in groups:
        N += len(group)
    #run through left and right    
    for group in groups:
        temp = 0
        if len(group) ==0:
            continue
        #count how many certain class in the group and divide to the group size to get the proportion
        for cla in classes: 
            count = 0
            for data in group:
                if data[-1] == cla:
                    count +=1
            por = float(count)/float(len(group))

            temp += por**2
        #get the gini index
        gini += (1 - temp) * len(group)/N
    return gini
            


# In[173]:

def split(dataset, classes):
    #store the which attribute to split, what is the treshold, left group & right group, gini number
    split_attri, split_value, split_groups, bestgini = 0, 0, [], 10000
    #decide which attribute to split
    for i in range(len(dataset[0])-1):       
        #setting the treshold by value in data for each attribute
        for data in dataset:
            left_split = []
            right_split = []
            treshold = data[i]
            #divide the data to left and right based on treshold
            for data in dataset:            
                if data[i] > treshold:
                    right_split.append(data)
                else:
                    left_split.append(data)
            groups = [left_split,right_split]
            #calculate the gini index
            gini = gini_calculator(groups,classes)
            
            #find out the lowest gini index to be our spliting attribute
            if gini < bestgini:
                split_attri, split_value, split_groups, bestgini = i ,data[i], groups, gini
    return {'split_attri': split_attri,'split_value':split_value, 'split_groups':split_groups,'bestgini':bestgini }


# In[174]:

def build_tree(dataset, classes):
    
    root_node = split(dataset, classes)
    node_split(root_node, classes)
    
    return root_node


# In[175]:

from collections import Counter
def terminal_node(group):
    results = []
    for data in group:
        results.append(data[-1])
    results = Counter(results)
    return results.most_common()[0][0]


# In[176]:

def checkEqual(group):
    results = []
    for data in group:
        results.append(data[-1])
    return results[1:] == results[:-1]


# In[177]:

def checkIDlist(group):
    for i in range(len(group)-1):
        if group[i][:-1] != group[i+1][:-1]:
            return False
    return True
        
        


# In[178]:

def node_split(node, classes):
    left_split, right_split = node['split_groups']

    if len(left_split) == 0:
        node['right_split'] = terminal_node(right_split)
        return
    if len(right_split) == 0:
        node['left_split'] = terminal_node(left_split)
        return

    if checkEqual(left_split) :
        node['left_split'] = terminal_node(left_split) 
    else:
        node['left_split'] = split(left_split,classes) 
        node_split(node['left_split'],classes)      
 
    if checkEqual(right_split):
        node['right_split']= terminal_node(right_split)        
    else:
        node['right_split'] = split(right_split,classes)
        node_split(node['right_split'],classes)   
    
    if checkEqual(left_split) and checkEqual(right_split):
        return node

    if checkIDlist(left_split):
        node['left_split'] = terminal_node(left_split)
    else:
        node['left_split'] = split(left_split,classes)
        node_split(node['left_split'],classes) 
        
    if checkIDlist(right_split):
        node['right_split'] = terminal_node(right_split)
    else:
        node['right_split'] = split(right_split,classes)
        node_split(node['right_split'],classes)           
    
    return node


# In[179]:

def get_classes(dataset):
    classes = []
    for data in dataset:
        classes.append(data[-1])
    return list(set(classes))


# In[180]:

def turn_list(df):
    ls = []
    for index, row in df.iterrows():
        ls.append(list(row))
    return ls


# In[181]:

def predict(node, data):
    
    if data[node['split_attri']] < node['split_value']:
        if isinstance(node['left_split'], dict):
            return predict(node['left_split'], data)
        else:
            return node['left_split']
    else:
        if isinstance(node['right_split'], dict):
            return predict(node['right_split'], data)
        else:
            return node['right_split']


# In[182]:

def decision_tree(traindata, testdata):
   tree = build_tree(traindata,get_classes(dataset))
   predictions = list()
   for data in testdata:
       prediction = predict(tree, data)
       predictions.append(prediction)
   return(predictions)


# In[183]:

def evaluate(traindata,testdata):
    test = []
    for data in testdata:
        test.append(data[-1])
    predictions = decision_tree(traindata, testdata)
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == test[i]:
            count +=1

    print 'accuacy:', float(count)/float(len(predictions))*100,'%' 


# In[184]:

import pandas as pd
df = pd.read_excel('/Users/eddiewu/downloads/immunotherapy.xlsx',header = 0, index_col = None)
dataset = turn_list(df)
trainlenth = int(len(dataset)*0.8)
traindata = dataset[:trainlenth]
testdata = dataset[trainlenth:]
evaluate(traindata,testdata)

