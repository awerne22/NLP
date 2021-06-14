
###### NOTE ######

"""

rebild configuration panel add flunctuacion analyze


 
try new input 
make better short 

+ spiner



2. add fmin  +
3. fix analyze for n>1(hcekc fa) + add condition +
4. add graph for fa +


5. add log po yaxec +
6.add rank and others +
7.add markov chain graph
8.chekc R  +-
9. check what is wrong with distribution graph +


upgrade save data button

add save selected item

make df like so

w | ∆f | fit |
20| 2  | 1.8 |
...

"""



import unicodedata




#with open("corpus/lotr_en.txt") as f:
#    file=f.read()
from time import asctime

from dash_core_components.Graph import Graph
from dash_html_components.Legend import Legend
from libs import *
#print(file)
def remove_punctuation(data):
    temp=[]
    for i in data:
        if i in punctuation:
            
            if i == '\xa0': #or i == '-'or i==' ':
                #temp.append(i.lower())
                continue
        else:
            temp.append(i.lower())
    return "".join(temp)
#data=remove_punctuation(file)
#print(data.split())

 


model=None





class Ngram (dict):
    def __init__ (self, iterable=None): # Ініціалізували наш розподіл як новий об'єкт класу, додаємо наявні елементи
        super(Ngram, self).__init__()
        self.F_i = 0 # число унікальних ключів в розподілі
        self.fa={}
        self.counts={}
        self.sums={}
        self.win={}
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
    
def make_dataframe(model,L,fmin=0):

    filtred_data=list(filter(lambda x:len(model[x].pos) >=fmin,model))
    if 'new_ngram' not in filtred_data:
        filtred_data.append("new_ngram")
    data={"rank":np.empty(len(filtred_data)),
          "ngram":[],
          "ƒ":np.empty(len(filtred_data),dtype=np.dtype(int))}

    #print(data['ngram'][0])
    for i,ngram in enumerate(filtred_data):
        #if ngram.__class__ is tuple:
        #    data["ngram"].append("  ".join(ngram))
        #else:
        data["ngram"].append(ngram)

        if ngram=="new_ngram":
            data['ƒ'][i]=sum(model[ngram].bool)
            continue
        data["ƒ"][i]=len(model[ngram].pos)
        
        #data["f_i"][i]=round(model[ngram].F_i/L,7)
    return pd.DataFrame(data=data)
#L=0#
#V=0#

def make_markov_chain(data,order=1,split='word'):
    global model,L,V
    model=dict()

    L=len(data)-order
    model['new_ngram']=Ngram()
    model['new_ngram'].bool=np.zeros(L,dtype=np.uint8)
    model['new_ngram'].pos=[]
    if order>1:
        for i in range(L):
            window = tuple (data[i: i+order])  # Додаємо в словник
            if window in model: # Приєднуємо до вже існуючого розподілу
                 model[window].update ([data[i+order]])
                 model[window].pos.append(i)
                 model[window].bool[i]=1

            else:
                 model[window] = Ngram ([data[i+order]])
                 model[window].pos=[]
                 model[window].pos.append(i)

                 model[window].bool=np.zeros(L,dtype=np.uint8)
                 model[window].bool[i]=1
                 model['new_ngram'].bool[i]=1
                 model['new_ngram'].pos.append(i+1)

    else:
        for i in range(L):
            if data[i] in model: # Приєднуємо до вже існуючого розподілу
                 model[data[i]].update ([data[i+order]])
                 model[data[i]].pos.append(i+1)
                 model[data[i]].bool[i]=1
            else:
                 model[data[i]] = Ngram ([data[i+order]])
                 model[data[i]].pos=[]
                 model[data[i]].pos.append(i)
                 model[data[i]].bool=np.zeros(L,dtype=np.uint8)
                 model[data[i]].bool[i]=1
                 model['new_ngram'].bool[i]=1
                 model['new_ngram'].pos.append(i+1)
    V=len(model)


from numba import types
from numba.typed import Dict
#@jit(nopython=True)
def calculate_distance(positions,L,option):
    if option=="no":
        return nbc(positions)
    if option=="ordinary":
        return obc(positions,L)
    if option=="periodic":
        return pbc(positions,L)


#@jit(nopython=True)
def nbc(positions):
    number_of_pos=len(positions)
    if number_of_pos ==1:
        return positions
    dt=np.empty(number_of_pos-1)
    for i in range(number_of_pos-1):
        dt[i]=(positions[i+1]-positions[i])-1
    return dt

#@jit(nopython=True)
def obc(positions,L):
    number_of_pos=len(positions)
    dt=np.empty(number_of_pos+1)
    dt[0]=positions[0]
    for i in range(number_of_pos-1):
        dt[i+1]=positions[i+1]-positions[i]
    dt[-1]=L-positions[-1]
    return dt

#@jit(nopython=True)
def pbc(positions,L):
    number_of_pos=len(positions)
    dt=np.empty(number_of_pos)
    for i in range(number_of_pos-1):
        dt[i]=positions[i+1]-positions[i]
    dt[-1]=L-positions[-1]+L+positions[0]
    return dt

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

#@jit(nopython=True)
def R(x):
    if len(x)==1:
        return 0
    avg=np.mean(x)
    avgs=np.mean(x**2)
    skv=np.sqrt(avgs-(avg**2))
    return skv/avg
#@jit(nopython=True)
def make_windows(x,args):
    wi,l,wsh=args
#    count=np.empty(len(range(0,l-wi,wsh)))#,dtype=np.uint8)
    count={}
   # di=dict()
    for index,i in enumerate(range(0,l-wi,wsh)):
        count[i]=x[i:i+wi]        #di[i+wi]=[x[i:i+wi]]
    return count
def calc_sum(x):
    sums=np.empty(len(x))
    for i,w in enumerate(x):
        sums[i]=np.sum(w)
    return sums
    

@jit(nopython=True)
def fit(x,a,b):
    return a*x**b


def prepere_data(data,n,split):
    global L
    if n is None:
        return dash.no_update
    temp_data=[]
    if n==1:
        if split=="word":
        #data.replace(" ","space")
            data=data.split()
            L=len(data)-n
            return data
        if split=='letter':
            data=remove_punctuation(data)
            for i in data:
                for j in i:
                    if j ==" ":
                        continue
                    temp_data.append(j)
            L=len(temp_data)-n
            return temp_data
        if split=='symbol':
            for i in data:
                for j in i:
                    if j ==" ":
                        temp_data.append("space")
                        continue
                    temp_data.append(j)
            L=len(temp_data)-n
            return temp_data

    if n>1:
        
        if split=="word":
            data=data.split()
            L=len(data)-n
            for i in range(L):
                window = tuple(data[i: i+n])
                temp_data.append(window)
            return temp_data
        if split=="letter":
            data=remove_punctuation(data.split())
            for i in data:
                for j in i:
                    temp_data.append(j)
            L=len(temp_data)-n
            data=temp_data
            temp_data=[]
            for i in range(L):
                window = tuple (data[i: i+n])
                temp_data.append(window)
            return temp_data
        if split=='symbol':
            temp_data=[]
            for i in data:
                for j in i:
                    if j==" ":
                        temp_data.append("space")
                        continue
                    temp_data.append(j)
            data=temp_data
            temp_data=[]
            L=len(data)-n
            for i in range(L):
                window=tuple(data[i:i+n])
                temp_data.append(window)
            return temp_data

#@jit(nopython=True)
def dfa(data,args):
    wi,wh,l=args
    count=np.empty(len(range(wi,l,wh)),dtype=np.uint8)
    for index,i in enumerate(range(0,l-wi,wh)):
        temp_v=[]
        x=[]
        for ngram in data[i:i+wi]:
            if ngram in temp_v:
                x.append(0)
            else:
                temp_v.append(ngram)
                x.append(1)
        count[index]=s(np.array(x,dtype=np.uint8))
        return count,mse(count)
class newNgram():
    def __init__(self,data,wh,l):
        self.data=data
        self.count={}
        self.dfa={}
        self.wh,self.l=wh,l
    def func(self,w):
        self.count[w],self.dfa[w]=dfa(self.data,(w,self.wh,self.l))
    


import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from os import listdir
import plotly.graph_objects as go

app =dash.Dash(__name__)

corpuses=listdir("corpus/")
colors={
    "background":"#a1a1a1",
    "text":"#a1a1a1"}

import dash_bootstrap_components as dbc
layout2=html.Div()

# old layout was fun but not what i wanted
#
layout1=html.Div([
                dbc.Row(
                    [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Configuration:"),
                                dbc.CardBody(
                                [
                                    html.Label("Choose file:"),
                                    html.Div(
                                        [
                                    dcc.Dropdown(id="corpus",options=[{"label":i,"value":i}for i in corpuses]),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupAddon("Size of ngram", addon_type="prepend"),
                                            dbc.Input(id="n_size",type="number",value=1),
                                        ],size="md",className="config"
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupAddon("Split by", addon_type="prepend"),
                                            dbc.Select(
                                                id="split",
                                                options=[
                                                    {"label":"symbol","value":"symbol"},
                                                    {"label":"word","value":"word"},
                                                    {"label":"letter","value":"letter"}
                                                   
                                                ],
                                                value="word"
                                            )
                                        ],size="md",className="config"
                                    ),

                                    dbc.InputGroup(
                                        [
                                            #dbc.InputGroupAddon("Boundary Condition:", addon_type="append"),
                                            dbc.Select(
                                                id="condition",
                                                options=[
                                                    {"label":"no","value":"no"},
                                                    {"label":"periodic","value":"periodic"},
                                                    {"label":"ordinary","value":"ordinary"}
                                                ],
                                                value="no"
                                            ),
                                            dbc.InputGroupAddon("Boundary Condition:", addon_type="append"),
                                        ],size="md",className="config"
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupAddon("filter",addon_type="prepend"),
                                            dbc.Input(id="f_min",type="number",value=0)
                                        ]
                                    ),
                                    html.Label("Sliding window"),

                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupAddon("Min window", addon_type="prepend"),
                                            dbc.Input(id="w",type="number"),
                                        ],size="md",className="window"
                                    ),
                                    
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupAddon("Window shift", addon_type="prepend"),
                                            dbc.Input(id="wh",type="number"),
                                        ],size="md",className="window"
                                    ),

                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupAddon("Window exspansion", addon_type="prepend"),
                                            dbc.Input(id="we",type="number"),
                                        ],size="md",className="window"
                                    ),

                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupAddon("Max window", addon_type="prepend"),
                                            dbc.Input(id="wm",type="number"),
                                        ],size="md",className="window"
                                    ),

                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="def",
                                                options=[
                                                    {"label":"static","value":"static"},
                                                    {"label":"dynamic","value":"dynamic"}                                                     
                                                ],
                                                value="static"
                                            ),
                                            dbc.InputGroupAddon("Definition", addon_type="append")
                                        ],size="md",className="window"
                                    ),

                                
                                   html.Br(),
                                            dbc.Button("Analyze", id="chain_button",color="primary",block=True),
                                            
                                            dbc.Button("Save data",id="save",color="danger",block=True),
                                            html.Div(id="temp_seve",
                                             children=[]
                                            )
                                        ]),
                                    html.Div(id="alert",children=[])
                                                                        #html.H6("Boundary Condition:"),
                                    #dcc.RadioItems(id='condition',options=[{"label":"no","value":"no"},{"label":"periodic","value":"periodic"},{"label":"ordinary","value":"ordinary"}],value="words"),
                                ]


                                            ),
                                
                            ],color="light",style={"margin-left":"0px","margin-top":"10px",}
                                ),
                        width={"size":3,"offset":0}
                        ),
                    dbc.Col(
                            [
                            dbc.Card(
                                [
                                dbc.CardHeader(
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(label="DataTable",tab_id="data_table"),
                                            dbc.Tab(label="MarkovChain",tab_id="markov_chain")
                                        ],
                                        id="dataframe",
                                        card=True,
                                        active_tab="data_table"
                                             )

                                             ),
                                dbc.CardBody(
                                    [
                                        html.Div(id="box_tab",
                                                 style={"display":"none","height":"400px"},
                                                 children=[dbc.Spinner(dt.DataTable(
                                                    id="table",
                                                columns=[{"name":i,"id":i}for i in ['rank',"ngram","ƒ","R","a","b","goodness"]],
                                                style_data={'whiteSpace': 'auto','height': 'auto'},
                                                editable=False,
                                                filter_action="native",
                                                sort_action="native",
                                                page_size=50,
                                                fixed_rows={'headers': True},
                                                fixed_columns={'headers': True},
                                                style_cell={'whiteSpace': 'normal',
                                                             'height': 'auto',
                                                            "widht":"auto",
                                                            'textAlign': 'right',
                                                             "fontSize":15,
                                                            "font-family":"sans-serif"},#'minWidth': 40, 'width': 95, 'maxWidth': 95},
                                                     style_table={"height":"400px","minWidth":"500px", 'overflowY': 'auto',"overflowX":"none"}
                                                 ))]),
                                        html.Div(id="box_chain",
                                                 style={"display":"none"},
                                                 children=[dbc.Spinner(dcc.Graph(id="chain",style={"height":"400px"}))]),

                                dbc.CardHeader("Characteristics"),
                                dbc.CardBody(
                                    [
                                        html.Div(["Length: "],id="l"),
                                        html.Div(["Vocabulary"],id="v"),
                                        html.Div(["TIme: "],id="t")
                                    ]
                                            )


                                     ]
                                             )
                                ],style={"padding":"0","margin-right":"0px","margin-top":"10px","height":"650px"}),
                            ],
                    width={"size":9,"padding":0}
                            ),
                                        ]
                        ),
                        dbc.Row([
                            dbc.Col(
                                width={"size":6,"offset":0},
                                children=[
                                    dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            dbc.Tabs(
                                                [
                                                    dbc.Tab(label="distribution",tab_id="tab1"),
                                                ],
                                                id='card-tabs1',
                                                card=True,
                                                active_tab="tab1"
                                            )
                                        ),
                                        dbc.CardBody([
                                           dcc.Graph(id="graphs")

                                        ])

                                    ],style={"height":"100%","widht":"100%","margin-right":"0%","margin-top":"10px","margin-left":"0%"}
                                )
                                ]                           ),
                            dbc.Col(
                                width={"size":6},
                                 children=[

                                     dbc.Card(
                                        [
                                        dbc.CardHeader(
                                            dbc.Tabs(
                                                [
                                                    dbc.Tab(label="flunctuation",tab_id="tab2"),
                                                    dbc.Tab(label="b/R",tab_id="tab3")
                                                ],
                                                id='card-tabs',
                                                card=True,
                                                active_tab="tab2"
                                            )
                                        ),
                                        dbc.CardBody([
                                            dcc.RadioItems(
                                                id="scale",
                                                options=[
                                                    {"label":"linear","value":"linear"},
                                                    {"label":"log","value":"log"}
                                                ],
                                                value="linear"

                                            ),
                                            dcc.Graph(id="fa")

                                        ])

                                        ],style={"height":"100%","widht":"100%","padding":"0","margin-right":"0%","margin-top":"10px","margin-left":"0%"}
                                )

                                ]
                            )
                        ]

                        ),
                        dbc.Row(
                            children=[
                                html.Br(),
                                html.Br()
                            ]
                        )
                        

                        ])
from dash.dependencies import Input,Output,State
app.layout=layout1
df=None
g=None
import plotly.express as px
from sklearn.metrics import r2_score
import networkx as nx

@app.callback([Output("w","value"),
               Output("wh","value"),
              # Output("we","value"),
               Output("wm","value"),
               Output("l","children")],
               [Input("corpus","value"),Input("split","value"),
              Input("def","value"),Input("n_size","value")])
def calc_window(corpus,split,defenition,n):
    if corpus is None:
        return dash.no_update,dash.no_update,dash.no_update,dash.no_update
    global L,data
    
    with open("corpus/"+corpus) as f:
        file=f.read()
        file=unicodedata.normalize("NFKD",file)
    if defenition=="dynamic":
        data=prepere_data(file,n,split)
        wm=int(L/10)
        w=int(wm/10)
    else:


        temp=[]
        if split =="letter":
            data=remove_punctuation(file)
            for word in data:
                for i in word:
                    if i == ' ':
                        #temp.append("space")
                        continue
                    temp.append(i)
            #temp.replace(" ","space")
            data=temp
        if split =="symbol":
            data=file
            for i in data:
                if i ==" ":
                    temp.append("space")
                else:
                    temp.append(i)
            #temp.replace(" ","space")
            data=temp
            
        if split =="word":
            data=remove_punctuation(file)
            #data.replace(" ","space")
            data=data.split()
            


        
        L=len(data)-n
        wm=int(L/10)
        w=int(wm/10)
    return [w,w,wm,["Lenght: "+str(L)]]
new_ngram=None
@app.callback([Output("table","data"),Output("chain","figure"),
               Output("box_tab","style"),
               Output("box_chain","style"),
               Output("alert","children"),
               Output("v","children"),
               Output("t","children")],
              [Input("chain_button","n_clicks"),
               Input("dataframe","active_tab")],
              [State("corpus","value"),
               State("n_size","value"),
               State("split","value"),
               State("condition","value"),
               State("f_min","value"),
               State("w","value"),
               State("wh","value"),
               State("we","value"),
               State("wm","value"),
               State("def","value")
               ])
def update_table(n,dataframe,corpus,n_size,split,condition,f_min,w,wh,we,wm,defenition):
    
    
    global data,L,V,model,ngram,df,g,new_ngram
    if dataframe=="markov_chain":
        if model is None:
            
            return dash.no_update,dash.no_update,{"display":"none"},{"display":'inline'},dash.no_update,dash.no_update,dash.no_update

        if n is None:
            
            return dash.no_update,dash.no_update,{"display":"none"},{"display":'inline'},dash.no_update,dash.no_update,dash.no_update

        #add alert corpus if not selected
        if corpus is None:
            return dash.no_updata,dash.no_update,{"display":"none"},{"display":"inline"},
        dbc.Alert("Please choose corpus",color="danger",duration=2000,dismissable=False),dash.no_update,dash.no_update

        if new_ngram is not None:
            return 
       ## make markov chain graph ###
        g=nx.MultiGraph()
        temp={}

        for ngram in df['ngram']:
            if n_size>1:
                ngram=tuple(ngram.split())
            
            g.add_node(ngram)
            temp[ngram[0]]=ngram
  
        for node in g.nodes():
            if node[0]=="new_ngram":
                node='new_ngram'
            for i in model[node]:
                if i in temp:
                    g.add_edge(node,temp[i],weight=model[node][i])
         


        pos=nx.spring_layout(g)

        edge_x = []
        edge_y = []
        for edge in g.edges():
            x0,y0=pos[edge[0]]
            x1,y1=pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in g.nodes():
            x,y=pos[node]
            node_x.append(x)
            node_y.append(y)
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        node_adjacencies = []
        node_text = []

        for node, adjacencies in enumerate(g.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            if n_size>1:
                node_text.append('<b>'+" ".join(adjacencies[0])+"</b>"+'<br><br>connections='+str(len(adjacencies[1])))
                continue
            node_text.append("<b>"+"".join(adjacencies[0])+"</b>"+'<br><br>connections: '+str(len(adjacencies[1])))

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        
                       showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        annotations=[ dict(
                          
                            showarrow=True,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )




        return dash.no_update,fig,{"display":"none"},{"display":'inline'},dash.no_update,dash.no_update,dash.no_update
    if dataframe == "data_table":
    
        if n is None:
            
            return dash.no_update,dash.no_update,{"display":'inline'},{"display":"none"},dash.no_update,dash.no_update,dash.no_update

        #add alert corpus if not selected
        if corpus is None:
            return dash.no_updata,dash.no_update,{"display":"inline"},{"display":"none"},dbc.Alert("Please choose corpus",color="danger",duration=2000,dismissable=False),dash.no_update,dash.no_update

        if defenition=="dynamic":

            start=time()
            windows=list(range(w,wm,we))
            #2. create newNgram

            new_ngram=newNgram(data,wh,L)
            #print(data[0:10])
            #print(windows)
            #with ThreadPoolExecutor() as e:
            #    e.map(new_ngram.func,windows)
            for w in windows:
                new_ngram.func(w)
            #calculate coefs
            temp_v=[]
            temp_pos=[]
            temp_bool=np.zeros(len(data))
            for i,ngram in enumerate(data):
                if ngram not in temp_v:
                    temp_v.append(ngram)
                    temp_pos.append(i)
                    temp_bool[i]=1

            new_ngram.bool=temp_bool
            new_ngram.pos=temp_pos
            new_ngram.dt=calculate_distance(np.array(temp_pos,dtype=np.uint8),L,condition)
            new_ngram.R=round(R(new_ngram.dt),7)
            c,cov=curve_fit(fit,[*new_ngram.dfa.keys()],[*new_ngram.dfa.values()],method='lm',maxfev=5000)
            new_ngram.a=round(c[0],7)
            new_ngram.b=round(c[1],7)
            new_ngram.temp_dfa=[]
            for w in new_ngram.dfa.keys():
                    new_ngram.temp_dfa.append(fit(w,new_ngram.a,new_ngram.b))
            new_ngram.goodness=round(r2_score([*new_ngram.dfa.values()],new_ngram.temp_dfa),7)
            df=pd.DataFrame()
            df['rank']=[1]
            df['ngram']=['new_ngram']
            df["ƒ"]=[len(temp_pos)]
            df['R']=[new_ngram.R]
            df["a"]=[new_ngram.a]
            df["b"]=[new_ngram.b]
            df['goodness']=[new_ngram.goodness]
            V=len(temp_v)








            #add df 



            
        
        else:
            #:print(model)
            if model is not None:
                return [df.to_dict("record"),dash.no_update,{"display":"inline"},{"display":"none"},
                        dash.no_update,dash.no_update,dash.no_update]
            ###  MAKE MARKOV CHAIN ####
            start=time()
            make_markov_chain(data,order=n_size,split=split)
            df=make_dataframe(model,L,f_min)
            for index,ngram in enumerate(df['ngram']):
                print(str(index)+" of "+str(len(df['ngram'])),end="\r")
                if ngram=="new_ngram":
                    model[ngram].dt=calculate_distance(model[ngram].pos,L,condition)
                    continue
                model[ngram].dt=calculate_distance(model[ngram].pos,L,condition)
            windows=list(range(w,wm,we))
            #fa(np.zeros(5),(1,3,4))
            #print(w,wm,we)
            #print(windows)
            def func(wind):
                wiwi=make_windows(model[ngram].bool,(wind,L,wh))
                sumas=calc_sum(wiwi.values())
                ffa=mse(sumas)
                return (wind,wiwi,sumas,ffa)
            #model[ngram].sums[wind],model[ngram].counts[wind],model[ngram].fa[wind]=fa(model[ngram].bool,(wind,L,wh))
            for index,ngram in enumerate(df['ngram']):
                print(str(index)+" of "+str(len(df['ngram'])),end="\r")

                with ThreadPoolExecutor() as e:
                    results=e.map(func,windows)
                for result in results:
                    #print(ngram,result[0],result[3])
                    model[ngram].win[result[0]]=result[1]
                    model[ngram].counts[result[0]]=result[2]
                    model[ngram].fa[result[0]]=result[3]


                        
                      # for i,ngram in enumerate(df["ngram"]):
           #     print(str(i)+"of"+str(len(df['ngram'])),end="\r")
            #    for wind in windows:
            #        func(wind)
            print("done calculation")
            #calculate_fa(df,model,w,wh,w,wm,L,condition)
            ###
            temp_b=[]

            temp_R=[]
            temp_error=[]
            temp_ngram=[]
            
                
            temp_a=[]
            
            for ngram in df['ngram']:
                model[ngram].temp_fa=[]
                ff=[*model[ngram].fa.values()]
                ww=[*model[ngram].fa.keys()]
                                
               
                c,cov=curve_fit(fit,ww,ff,method='lm',maxfev=5000)
                model[ngram].a=c[0]
                model[ngram].b=c[1]
                for w in ww:
                    model[ngram].temp_fa.append(fit(w,model[ngram].a,model[ngram].b))
                temp_error.append(round(r2_score(ff,model[ngram].temp_fa),5))
                temp_b.append(round(c[1],7))
                temp_a.append(round(c[0],7))

                
                if ngram.__class__ is tuple:

                    temp_ngram.append(" ".join(ngram))
                #print(np.array(model[ngram].dt))
                r=round(R(model[ngram].dt),7)
                temp_R.append(r)
                model[ngram].R=r

            if n_size>1:
                temp_ngram.append("new_ngram")
                df["ngram"]=temp_ngram
            df['R']=temp_R
            df['b']=temp_b 
            df['a']=temp_a
            df['goodness']=temp_error

            df=df.sort_values(by="ƒ",ascending=False)
            df['rank']=range(1,len(temp_R)+1)
            df=df.set_index(pd.Index(np.arange(len(df))))
            #print(df)
            
        #table.data=df.to_dict("records")
        return [df.to_dict("record"),dash.no_update,{"display":"inline"},{"display":"none"},dash.no_update,
                ["Vocabulary: "+str(V)],["Time:"+str(round(time()-start,4))]]
                #dash.no_update,["Vocabulary: ",V],["Time: ",round(time()-start,6)]]
clikced_ngram=None
#@app.callback([Output("temp_seve","children")],
#              [Input("graphs","clickData")])
#def check_dist(clicked_data):
 #   if clicked_data:
 #       fig.add_trace(go.Scatter(x=np.arange(L),y=model[ngram].bool))
  #      if fa_click:
   #     fig.add_trace(go.Bar(x=np.arange(wh,L,wh),y=model[ngram].counts[fa_click["points"][0]["x"]],name="‚àë‚àÜw"))

 #   return dash.no_update
@app.callback([Output("graphs","figure"),Output("fa","figure"),],
              [Input("dataframe","active_tab"),
                  Input("card-tabs","active_tab"),
                Input("table","active_cell"),
               Input("table","derived_virtual_selected_rows"),
               Input("table","derived_virtual_indices"),
               Input("chain","clickData"),
               Input("scale","value"),
               Input("fa","clickData"),
               Input("graphs","clickData"),
               Input("wh","value")],
              [State("n_size","value"),
               State("def","value"),])
def tab_content(active_tab2,active_tab1,active_cell,row_ids,ids,clicked_data,scale,fa_click,graph_click,wh,n,defenition):
    global model,df,L,g,new_ngram
    if df is None:
        return dash.no_update,dash.no_update

    if ids is None:
        return dash.no_update,dash.no_update


    df=df.reindex(pd.Index(ids))
    fig=go.Figure()

    fig.update_layout(margin=dict(l=0,r=0,t=0,b=10))
    fig1=go.Figure()

    fig1.update_layout(margin=dict(l=0,r=0,t=0,b=15))
    #print(active_tab2)
    if active_tab2=="markov_chain":
        if defenition =="dynamic":
            return dash.no_update,dash.no_update
        if clicked_data:
            nodes=np.array(g.nodes())
            #print(nodes[clicked_data['points'][0]['pointNumber']])
            ngram=nodes[clicked_data['points'][0]['pointNumber']]
            if n>1:
                ngram=tuple(nodes[clicked_data['points'][0]['pointNumber']])
                if ngram[0]=='new_ngram':
                    ngram='new_ngram'
            if active_tab1=="tab2":
                fig.add_trace(go.Scatter(x=np.arange(L),y=model[ngram].bool))
                if fa_click:
                    fig.add_trace(go.Bar(x=np.arange(wh,L,wh),y=model[ngram].counts[fa_click["points"][0]["x"]],name="∑∆w"))
                fa_click=None
                fig1.add_trace(
                        go.Scatter(x=[*model[ngram].fa.keys()],
                                         y=[*model[ngram].fa.values()],
                                         mode='markers',
                                        name="∆F"))
                fig1.add_trace(go.Scatter(
                                    x=[*model[ngram].fa.keys()],
                                    y=model[ngram].temp_fa,
                                    name="fit"))
                fig1.update_xaxes(type=scale)
                fig1.update_yaxes(type=scale)
                fig1.update_layout(hovermode="x unified")
                #fig.update_layout(hovermode="x")

                return fig,fig1
            if active_tab1=="tab3":
                fig.add_trace(go.Scatter(x=np.arange(L),y=model[ngram].bool))
                if fa_click:
                    fig.add_trace(go.Bar(x=np.arange(wh,L,wh),y=model[ngram].counts[fa_click["points"][0]["x"]],name="∑∆w"))
                    print(model[ngram].sums[fa_click['points'][0]['x']])
                fa_click=None

                #fig.update_xaxes(type=scale)

                hover_data=[]                
                for data in df['ngram']:
                    hover_data.append("".join(data))
                fig1.add_trace(go.Scatter(x=df["R"],y=df["b"],mode="markers",text=hover_data))
                fig1.add_trace(go.Scatter(x=[model[ngram].R],
                                         y=[model[ngram].b],
                                         mode="markers",
                                         text=' '.join(ngram),
                                         marker=dict(
                                             size=20,
                                             color="red"
                                         ))) 
                fig1.update_layout(showlegend=False)
                fig1.update_yaxes(type=scale)
                fig1.update_xaxes(type=scale)
                #fig.update_layout(hovermode="x")
                fig1.update_layout(hovermode="x unified")
                return fig,fig1
            else:
                return fig,fig1





        return dash.no_update,dash.no_update
    else:
        if active_tab1=="tab2":
            if active_cell:
                
                if defenition =="dynamic":
                    fig.add_trace(go.Scatter(x=np.arange(L),y=new_ngram.bool))
                    
                    if fa_click:
                        #print(new_ngram)
                        fig.add_trace(go.Bar(x=np.arange(wh,L,wh),
                                             y=new_ngram.count[fa_click["points"][0]["x"]],name="∑∆w"))


                    fig1.add_trace(go.Scatter(x=[*new_ngram.dfa.keys()],y=[*new_ngram.dfa.values()],mode='markers',name="∆F"))
                    fig1.add_trace(go.Scatter(x=[*new_ngram.dfa.keys()],y=[*new_ngram.temp_dfa],name="fit=aw^b"))
                    fig1.update_xaxes(type=scale)
                    fig1.update_yaxes(type=scale)
                    fig1.update_layout(hovermode="x unified")
                    print(1)
                    return fig,fig1

                if n>1:
                    ngram=tuple(df['ngram'][ids[active_cell['row']]].split())
                    if ngram[0]=='new_ngram':
                        ngram='new_ngram'
                else:
                    ngram=df['ngram'][ids[active_cell['row']]]
                fig.add_trace(go.Scatter(x=np.arange(L),y=model[ngram].bool,name="positions"))
                print(ngram)
                if fa_click:
                    ww=fa_click['points'][0]["x"]
                    fig.add_trace(go.Bar(x=np.arange(0,L,wh),y=model[ngram].counts[ww],name="∑∆w"))
                if graph_click:
                    www=graph_click['points'][0]['x']
                graph_click=None
                fa_click=None


#                fig.add_trace(go.Scatter(x=np.arange(L),y=model[ngram].bool))
                #fig.update_xaxes(type=scale)
                temp_ww=[*model[ngram].fa.keys()]
                #print(temp_ww)
                #print(model[ngram].temp_fa)
                fig1.add_trace(
                        go.Scatter(x=temp_ww,
                                         y=[*model[ngram].fa.values()],
                                         mode='markers',
                                        name="∆F"))
                #print()
                fig1.add_trace(go.Scatter(
                                    x=temp_ww,
                                    y=model[ngram].temp_fa,
                                    name="fit=aw^b"))
                fig1.update_xaxes(type=scale)
                fig1.update_yaxes(type=scale)
                fig.update_layout(showlegend=False)
                fig1.update_layout(hovermode="x unified")
                #fig.update_layout(hovermode="x unified")
                
                return fig,fig1
            else:
                return fig,fig1
        else:
            hover_data=[]
            if active_cell:
                
                if defenition =="dynamic":
                    #ngram="new_ngram"
                    fig.add_trace(go.Scatter(x=np.arange(L),y=new_ngram.bool))
                    if fa_click:
                        fig.add_trace(go.Bar(x=np.arange(0,L,wh),y=new_ngram.count[fa_click["points"][0]["x"]],name="∑∆w"))


                    fig1.add_trace(go.Scatter(x=[new_ngram.R],y=[new_ngram.b],mode='markers',
                                              ))
                    fig1.update_xaxes(type=scale)
                    fig1.update_yaxes(type=scale)
                    #fig1.update_layout(hovermode="x unified")
                                
                    return fig,fig1

                if n>1:
                    ngram=tuple(df['ngram'][ids[active_cell['row']]].split())
                    if ngram[0]=='new_ngram':
                        ngram='new_ngram'
                else:
                    ngram=df['ngram'][ids[active_cell['row']]]
                print(ngram)
                for data in df['ngram']:
                    hover_data.append("".join(data))
                fig.add_trace(go.Scatter(x=np.arange(L),y=model[ngram].bool,name="positions"))
                if fa_click:
                    ww=fa_click['points'][0]["x"]
                    fig.add_trace(go.Bar(x=np.arange(ww,L,wh),y=model[ngram].counts[ww],name="∑∆w"))

                fa_click=None
                if graph_click:
                    pass
                    #print(model[ngram].sums.keys())
                    #print(model[ngram].sums[graph_click['points'][0]['x']])
                graph_click=None


                #fig.update_xaxes(type=scale)

                fig1.add_trace(go.Scatter(x=df["R"],y=df["b"],mode="markers",text=hover_data))
                fig1.add_trace(go.Scatter(x=[df['R'][active_cell['row']]],
                                         y=[df["b"][active_cell['row']]],
                                         mode="markers",
                                         text=' '.join(ngram),
                                         marker=dict(
                                             size=20,
                                             color="red"
                                         ))) 
                fig1.update_layout(showlegend=False)
                fig1.update_yaxes(type=scale)
                fig1.update_xaxes(type=scale)
                #fig1.update_layout(hovermode="x unified")
                #fig.update_layout(hovermode="x unified")

                return fig,fig1
            else:
                #fig.add_trace(go.Scatter(x=np.arange(L),y=model[ngram].bool))

                #fig.update_xaxes(type=scale)

                for data in df["ngram"]:
                    hover_data.append("".join(data))
                fig1.add_trace(go.Scatter(x=df["R"],y=df["b"],mode="markers",text=hover_data))
                fig1.update_yaxes(type=scale)
                fig1.update_xaxes(type=scale)
                #fig1.update_layout(hovermode="x unified")
                #fig.update_layout(hovermode="x unified")

            return fig,fig1

        return dash.no_update,dash.no_update
@app.callback([Output("temp_seve","children")],
              [Input("save","n_clicks"),
               Input("table","active_cell"),
               Input("table","derived_virtual_indices")],
              [State("corpus","value"),
               State("n_size","value"),
               State("w","value"),
               State("wh","value"),
               State("we","value"),
               State("wm","value"),
               State("f_min","value"),
               State("condition","value"),
               State("def","value")])
              
def save(n,active_cell,ids,file,n_size,w,wh,we,wm,fmin,opt,defenition):
    if n is None :
        return dash.no_update
    else:
        global df,model,new_ngram
        
        if defenition=="dynamic":
            writer=pd.ExcelWriter("saved_data/{0} contition={7},fmin={1},n={2},w=({3},{4},{5},{6}),defenition={8}.xlsx".format(file,fmin,n_size,w,wh,we,wm,opt,defenition))
            df.to_excel(writer)
            writer.save()
            if active_cell:
                writer=pd.ExcelWriter("saved_data/"+file+" new_ngram.xlsx")
                df1=pd.DataFrame()
                df1["w"]=[*new_ngram.dfa.keys()]
                df1['‚àÜF']=[*new_ngram.dfa.values()]
                df1['fit=a*w^b']=new_ngram.temp_dfa
                df1.to_excel(writer)
                writer.save()
            return dash.no_update





        writer=pd.ExcelWriter("saved_data/{0} contition={7},fmin={1},n={2},w=({3},{4},{5},{6}),defenition={8}.xlsx".format(file,fmin,n_size,w,wh,we,wm,opt,defenition))
        df.to_excel(writer)
        writer.save()
        if active_cell:
            ngram=df['ngram'][ids[active_cell['row']]]
            writer=pd.ExcelWriter("saved_data/"+file+" "+ngram+".xlsx")
            df1=pd.DataFrame()
            df1["w"]=[*model[ngram].fa.keys()]
            df1['∆F']=[*model[ngram].fa.values()]
            df1['fit=a*w^b']=model[ngram].temp_fa
            df1.to_excel(writer)
            writer.save()
    return dash.no_update


import webbrowser
if __name__=="__main__":
    #webbrowser.open("http://127.0.0.1:8050/")

    app.run_server(debug=True)

    #webbrowser.open("http://127.0.0.1:8050/")
    #main()





