import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import curve_fit
from memory_profiler import profile
import sys
#with open("corpus/lotr_en.txt") as f:
#    file=f.read()

#print(file)
from string import punctuation
def remove_punctuation(data):
    temp=[]
    for i in data:
        if i in punctuation:
            if i == "'" or i == '-'or i==' ':
                temp.append(i.lower())
        else:
            temp.append(i.lower())
    return "".join(temp)
#data=remove_punctuation(file)
#print(data.split())



class Ngram (dict):
    def __init__ (self, iterable=None): # Ініціалізували наш розподіл як новий об'єкт класу, додаємо наявні елементи
        super(Ngram, self).__init__()
        self.F_i = 0 # число унікальних ключів в розподілі
        self.fa={}
        self.counts={}
        if iterable:
           self.update (iterable)
    def update (self, iterable): # Оновлюємо розподіл елементами з наявного итерируемого набору даних
        for item in iterable:
           if item in self:
               self[item]+=1
           else:
                self[item] = 1
                self.F_i+=1
    def hist(self):
        plt.bar(self.keys(),self.values())
        plt.show()

def make_dataframe(model,fmin):
    filtred_data=list(filter(lambda x:model[x].F_i >=fmin,model))
    data={"ngram":[],
          "F_i":np.empty(len(filtred_data),dtype=np.dtype(int))}

    #print(data['ngram'][0])
    for i,ngram in enumerate(filtred_data):
        data["ngram"].append(ngram)
        data["F_i"][i]=model[ngram].F_i
    return pd.DataFrame(data=data)
#L=0#
#V=0#
def make_markov_chain(data,order=1,split='word'):
    global model,L,V
    model=dict()

    if split=="symbol":
        temp=[]
        for word in data:
            for symbol in word:
                temp.append(symbol)
        data=temp
    L=len(data)-order
    if order>1:
        for i in range(L):
            window = tuple (data[i: i+order])  # Додаємо в словник
            if window in model: # Приєднуємо до вже існуючого розподілу
                 model[window].update ([data[i+order]])
                 model[window].pos.append(i+1)
                 model[window].bool[i]=1

            else:
                 model[window] = Ngram ([data[i+order]])
                 model[window].pos=[]
                 model[window].pos.append(i+i)
                 model[window].bool=np.zeros(len(data)-order)
                 model[window].bool[i]=1
    else:
        for i in range(L):
            if data[i] in model: # Приєднуємо до вже існуючого розподілу
                 model[data[i]].update ([data[i+order]])
                 model[data[i]].pos.append(i+order)
                 model[data[i]].bool[i]=1
            else:
                 model[data[i]] = Ngram ([data[i+order]])
                 model[data[i]].pos=[]
                 model[data[i]].pos.append(i+1)
                 model[data[i]].bool=np.zeros(len(data)-order)
                 model[data[i]].bool[i]=1
    V=len(model)

#print(df["ngram"])
@jit(nopython=True)
def s(window):
    suma=0
    for i in range(len(window)):
        suma+=window[i]
    return suma
@jit(nopython=True)
def mse(x):
    t=np.mean(x)
    st=np.mean(x**2)
    return np.sqrt(st-(t**2))

@jit(nopython=True)
def R(x):
    t=np.mean(x)
    ts=np.mean(x**2)
    return np.sqrt(ts-(t**2))/t
@jit(nopython=True)
def fa(x,args):
    #print(w)
    wi,wsh,l=args
    #print(wi,wsh,l)
    count=np.empty(len(range(wi,l,wsh)),dtype=np.uint8)

    for index,i in enumerate(range(0,l-wi,wsh)):
        count[index]=s(x[i:i+wi])

    return count,mse(count)
@jit(nopython=True)
def fit(x,a,b):
    return a*x**b




"""
def secound_pool(ngram):
    return fa(model[ngram].bool,(temp_w,wmax,we,wh,L))

def first_pool(wind):
    #print(w)
    #print(list(df['ngram']))
    #print(model['entropy'])
    global temp_w
    temp_w=wind
    with ThreadPoolExecutor() as j:
        result = j.map(secound_pool,df['ngram'])
    for f,ngram in zip(result,df['ngram']):
        model[ngram].counts[wind]=f[0]
        model[ngram].fa[wind]=f[1]


with ThreadPoolExecutor() as e:
    windows=list(range(w,wmax,we))
    e.map(first_pool,windows)
"""
def calculate_fa(df,model,*args):
    fa(np.zeros(5),(1,2,3))
    w,wmax,we,wh,L=args
    def func(window):
        model[ngram].counts[window],model[ngram].fa[window]=fa(model[ngram].bool,(window,wh,L))
    for index,ngram in enumerate(df['ngram']):
        print(str(index)+" of "+str(len(df['ngram'])),end="\r")
        with ThreadPoolExecutor() as e:
            wi=list(range(w,wmax,we))
            e.map(func,wi)


    pass

#print("fa time:",time()-start)
#print(df)
L,V=0,0
wh=0
model=0
ngram=0
@profile
def main():
    global L,V,wh,model,ngram
    with open("corpus/lotr_en.txt") as f:
        file=f.read()
    data=remove_punctuation(file)
    start=time()
    fmin=100
    order=2
    split="word"
    make_markov_chain(data.split(),order=order,split=split)

    #model
    df=make_dataframe(model,fmin)
    print("chain time:",time()-start)
    #print(df)

    ### CALCULATE FA ###

    print("order:",order)
    print("split by:",split)
    print("L:",L)
    print("V: ",V)
    print("fmin:",fmin)
    print("valid V:",len(df['ngram']))
    print()
    wmax=int(L/20)
    w=int(wmax/10)
    we=int(wmax/10)
    wh=w
    print("w:",w)
    print("wmax:",wmax)
    print("we:",we)
    print("wh:",wh)
    #model['entropy'].bool



    start=time()
    temp_w=0
#    g()
 #   input()

    calculate_fa(df,model,w,wmax,we,wh,L)
    temp_b=[]
    temp_fi=[]
    temp_R=[]
    print(df)
    print()
    #for ngram in model:
        #print(model[ngram].fa)



    for ngram in df['ngram']:
        c,cov=curve_fit(fit,[*model[ngram].fa.keys()],[*model[ngram].fa.values()],maxfev=5000)
        model[ngram].a=c[0]
        model[ngram].b=c[1]
        temp_b.append(c[1])
        temp_fi.append(model[ngram].F_i/L)
        temp_R.append(R(np.array(model[ngram].pos)))
    df['R']=temp_R
    df['f_i']=temp_fi
    df['alpha']=temp_b
    print(df)



    pass
if __name__=="__main__":
    main()



























