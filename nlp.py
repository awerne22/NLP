#with open("corpus/lotr_en.txt") as f:
#    file=f.read()
from libs import *
#print(file)
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
                 model[window].pos.append(i)
                 #model[window].bool.append(i)

            else:
                 model[window] = Ngram ([data[i+order]])
                 model[window].pos=[]
                 model[window].pos.append(i)

                 #model[window].bool=[]#np.zeros(len(data)-order)
                 #model[window].bool.append(i)
    else:
        for i in range(L):
            if data[i] in model: # Приєднуємо до вже існуючого розподілу
                 model[data[i]].update ([data[i+order]])
                 model[data[i]].pos.append(i)
                 #model[data[i]].bool.append(i)
            else:
                 model[data[i]] = Ngram ([data[i+order]])
                 model[data[i]].pos=[]
                 model[data[i]].pos.append(i)
                 #model[data[i]].bool=[]#np.zeros(len(data)-order)
                 #model[data[i]].bool.append(i)
    V=len(model)



#@jit(nopython=True)
def calculate_distance(positions,L,option):
    if option=="no":
        return nbc(positions)
    if option=="ordinary":
        return obc(positions,L)
    if option=="periodic":
        return pbc(positions,L)


@jit(nopython=True)
def nbc(positions):
    number_of_pos=len(positions)
    if number_of_pos ==1:
        return positions
    dt=np.empty(number_of_pos-1,dtype=np.uint8)
    for i in range(number_of_pos-1):
        dt[i]=positions[i+1]-positions[i]
    return dt

@jit(nopython=True)
def obc(positions,L):
    number_of_pos=len(positions)
    dt=np.empty(number_of_pos+1,dtype=np.uint8)
    dt[0]=positions[0]
    for i in range(number_of_pos-1):
        dt[i+1]=positions[i+1]-positions[i]
    dt[-1]=L-positions[-1]
    return dt

@jit(nopython=True)
def pbc(positions,L):
    number_of_pos=len(positions)
    dt=np.empty(number_of_pos,dtype=np.uint8)
    for i in range(number_of_pos-1):
        dt[i]=positions[i+1]-positions[i]
    dt[-1]=L-positions[-1]+L+positions[0]
    return dt

#print(df["ngram"])
@jit(nopython=True)
def s(*args):
    indexes,w,wm=args
    suma=0
    for index in indexes:
        if index >=w and index <=wm:
            suma+=1
        if index >wm:
            return suma
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
    wi,whi,l=args
    #print(wi,wsh,l)
    count=np.empty(len(range(wi,l,whi)),dtype=np.uint8)
    for index,i in enumerate(range(0,l-wi,whi)):
        count[index]=s(x,i,i+wi)

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
    fa(np.array([1,2],dtype=np.uint8),(1,2,3))
    w,wmax,we,wh,L,opt=args
    def func(window):
        dt=calculate_distance(np.array(model[ngram].pos,dtype=np.uint8),L,opt)
        model[ngram].counts[window],model[ngram].fa[window]=fa(dt,(window,wh,L))
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
    fmin=4
    order=1
    split="word"
    option="obc"
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
    print("option:",option)
    #model['entropy'].bool



    start=time()
    temp_w=0
#    g()
 #   input()

    calculate_fa(df,model,w,wmax,we,wh,L,option)
    temp_b=[]
    temp_fi=[]
    temp_R=[]
#    print(df)
    print()
    print("fa time:",time()-start)
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

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from os import listdir

#external_stylesheets = ['stylesheets.css']


app =dash.Dash(__name__)

corpus=listdir("corpus/")
colors={
    "background":"#a1a1a1",
    "text":"#a1a1a1"}

app.layout=html.Div(style={"backgroundColor":colors["background"]},
                    children=[html.Div(style={"backgroundColor":colors["text"],
                                                "widht":"20%",
                                              "float":"left",
                                              "text-align":"center",
                                              "padding":"1%"},
                              children=[
                                        html.Div(children=[html.Label("Choose corpus"),
                                        dcc.Dropdown(
                                            id="corpus",
                                            options=[{"label":i,"value":i} for i in corpus],
                                        )],style={"width":"90%",
                                                  "margin-right":"5%",
                                                  "margin-left":"5%"}),
                                        html.Div(children=[
                                        html.Label("size of ngram"),
                                        dcc.Slider(
                                            id="n-size",
                                            min=1,
                                            max=7,
                                            marks={i:"{}".format(i)for i in range(1,8)},
                                            value=1)],style={"width":"80%",
                                                             "margin-left":"10%",
                                                             "margin-right":"10%"}),
                                        html.Div(children=[
                                        html.Label("split by"),
                                        dcc.Dropdown(
                                            id="split",
                                            options=[{"label":i,"value":i} for i in ["word","symbol"]],
                                            value="word")],style={"width":"60%",
                                                                  "margin-left":"20%",
                                                                  "margin-right":"20%"}),
                                        html.Div(children=[
                                         html.Label("Boundary Condition"),
                                        dcc.Dropdown(
                                            id="options",
                                            options=[{"label":i,"value":i} for i in ["no","periodic","ordinary"]],
                                            value="no")],style={"width":"60%",
                                                                "margin-left":"20%",
                                                                "margin-right":"20%"}),
                                        html.Div(children=[
                                        html.Label("freauency of ngram"),
                                        dcc.Input(
                                            id="fmin",
                                            style={"padding":"5%","width":"50%"},
                                            placeholder="filter",
                                            type='number',
                                            debounce=True,
                                            value="")],style={"width":'60%',
                                                            "margin-left":"19%",
                                                             "margin-right":"19%"})

                                        ]),
                              html.Div(style={"float":"right","width":"75%"},
                                       children=[dt.DataTable(
                                       id='table',
                                           columns=[{"name":i,"id":i}for i in ["ngram","F_i","f_i","R","alpha"]],
                                       data=[])])])
from dash.dependencies import Input,Output,State
@app.callback(Output("table","data"),
              [Input("corpus","value"),
               Input("n-size","value"),
               Input("split","value"),
               Input("options","value"),
               Input("fmin","value")])
def update_table(corpus,n_size,split,options,fmin):
    print(corpus)
    print(n_size)
    print(split)
    print(options)
    print(fmin)
    if corpus is None:
        pass
    else:

    #global L,V,wh,model,ngram
    #with open("corpus/"+corpus) as f:
    #    file=f.read()
    #return [{"name":i,"id":i}for i in ["ngram","fmin"]]
    #data=remove_punctuation(file)
    #start=time()
    #fmin=4
    #order=1
    #split="word"
    #option="obc"
    #make_markov_chain(data.split(),order=n_size,split=split)
    #print()
    #model
    #df=make_dataframe(model,int(fmin))
    #return [{"name":i,"id":i}for i in df.columns]

    #print("chain time:",time()-start)
    #print(df)
    #return df.to_dict(),[{"name":i,"id":i}for i in df.columns]
    ### CALCULATE FA ###
        pass
import webbrowser
if __name__=="__main__":
    webbrowser.open("http://127.0.0.1:8050/")

    app.run_server(debug=True)

    #webbrowser.open("http://127.0.0.1:8050/")
    #main()



















