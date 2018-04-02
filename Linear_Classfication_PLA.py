
import copy
import random

def PLA(data,rand = False,nu=1):

    def inner_product(x,y):
        return sum(i*j for i,j in zip(x,y))

    data = copy.deepcopy(data)
    if rand:
        random.seed()
        random.shuffle(data)
    
    update = 0
    
    # init weight vector
    w = [0] * len(data[0][0])

    while True:       
        check = True
        for raw in data:
            x,y = raw
            r = inner_product(w,x)
            
            # The same sign: > 0
            if y*r <= 0:
                err = False
                update = update + 1
                w = [w + nu*x*y for w,x in zip(w,x)]
        if check == True:
            break

    return w,update


def hw15():

    f = open('./hw1_15_train.dat.txt','r')
    data = []
    for line in f:
        l = line.split()
        data.append([[float(i) for i in l[:-1]],int(l[-1])])
    f.close()

    print PLA(data)
    

def hw16():

    f = open('./hw1_15_train.dat.txt','r')
    data = []
    for line in f:
        l = line.split()
        data.append([[float(i) for i in l[:-1]],int(l[-1])])
    f.close()

    updates = []

    for i in range(0,2000):
        
        w,u = PLA(data,True)
        print '#%d' % (i),w,u
        updates.append(u)

    print 'Avg:',float(sum(updates))/len(updates)


def hw17():

    f = open('./hw1_15_train.dat.txt','r')
    data = []
    for line in f:
        l = line.split()
        data.append([[float(i) for i in l[:-1]],int(l[-1])])
    f.close()

    updates = []

    for i in range(0,2000):
        
        w,u = PLA(data,True,nu=0.5)
        print '#%d' % (i),w,u
        updates.append(u)

    print 'Avg:',float(sum(updates))/len(updates)



