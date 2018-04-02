
# coding: utf-8

# In[25]:


import copy
import random
def pocket(data,updatemax,nu =1):

    def inner_product(x,y):
        return sum(i*j for i,j in zip(x,y))

    data = copy.deepcopy(data)

    def error(w):
        e = 0
        for i in range(0,len(data)):
            x,y = data[i]
            if inner_product(w,x)*y <= 0:
                e = e +1
                
        return e

    update = 0
    
    # init weight vector
    w = [0] * len(data[0][0])
    w_best = w
    we = error(w)
    
    random.seed()
    while we != 0:       
        lenth = len(data)
        i = random.randint(1,lenth-1)
        
        x,y = data[i]
        r = inner_product(w,x)

        # The same sign: > 0
        if y*r <= 0:

            update = update + 1
            w_new = [w + nu*x*y for w,x in zip(w,x)]
#            print 'wnew',w_new
            w_new_e = error(w_new)
            if w_new_e < we:
                w_best = w_new
 #               print 'wbest',w_best
                we = error(w_new)
                w = w_new
 #               print 'w',w
            else:
                w = w_new
                    
        if update > updatemax:
            break

    return w_best,update,


# In[26]:


def hw17():

    f = open('./hw1_15_train.dat.txt','r')
    data = []
    for line in f:
        l = line.split()
        data.append([[float(i) for i in l[:-1]],int(l[-1])])
    f.close()

    updates = []

    for i in range(0,2000):
        
        w,u = pocket(data,50)
        print '#%d' % (i),w,u
        updates.append(u)

    print 'Avg:',float(sum(updates))/len(updates)


# In[27]:


hw17()


# In[11]:


f = open('./hw1_15_train.dat.txt','r')
data = []
for line in f:
    l = line.split()
    data.append([[float(i) for i in l[:-1]],int(l[-1])])
f.close()
len(data)

