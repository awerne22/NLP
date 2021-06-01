
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

"""



import unicodedata




#with open("corpus/lotr_en.txt") as f:
#    file=f.read()
from time import asctime

from dash_core_components.Graph import Graph
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
    
def make_dataframe(model,L,fmin=0):

    filtred_data=list(filter(lambda x:model[x].F_i >=fmin,model))
    data={"rank":np.empty(len(filtred_data)),
          "ngram":[],
          "ƒ":np.empty(len(filtred_data),dtype=np.dtype(int))}

    #print(data['ngram'][0])
    for i,ngram in enumerate(filtred_data):
        #if ngram.__class__ is tuple:
        #    data["ngram"].append("  ".join(ngram))
        #else:
        data["ngram"].append(ngram)
        data["ƒ"][i]=len(model[ngram].pos)
        
        #data["f_i"][i]=round(model[ngram].F_i/L,7)
    return pd.DataFrame(data=data)
#L=0#
#V=0#

def make_markov_chain(data,order=1,split='word'):
    global model,L,V
    model=dict()

    L=len(data)-order
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

                 model[window].bool=np.zeros(L)
                 model[window].bool[i]=1
    else:
        for i in range(L):
            if data[i] in model: # Приєднуємо до вже існуючого розподілу
                 model[data[i]].update ([data[i+order]])
                 model[data[i]].pos.append(i)
                 model[data[i]].bool[i]=1
            else:
                 model[data[i]] = Ngram ([data[i+order]])
                 model[data[i]].pos=[]
                 model[data[i]].pos.append(i)
                 model[data[i]].bool=np.zeros(L)
                 model[data[i]].bool[i]=1
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
    if len(x)==1:
        return 0.0
    t=np.mean(x)
    ts=np.mean(x**2)
    return np.sqrt(ts-(t**2))/t
@jit(nopython=True)
def fa(x,args):
    wi,wsh,l=args
    count=np.empty(len(range(wi,l,wsh)),dtype=np.uint8)
    for index,i in enumerate(range(0,l-wi,wsh)):
        count[index]=s(x[i:i+wi])
    return count,mse(count)
@jit(nopython=True)
def fit(x,a,b):
    return a*x**b





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


                                    #dbc.Input(placeholder="size of ngram",type="number"),
                                    #html.H6("Size of ngram:"),
                                    #dcc.Slider(id="n_size",min=1,max=9,value=1,marks={i:"{}".format(i)for i in range(1,10)}),
                                    #html.H6("Split by:"),
                                    #dcc.RadioItems(id='split',options=[{"label":"symbol","value":"symbol"},{"label":"word","value":"word"}],value="word"),
                                    #html.H6("Boundary Condition:"),
                                    #dcc.RadioItems(id='condition',options=[{"label":"no","value":"no"},{"label":"periodic","value":"periodic"},{"label":"ordinary","value":"ordinary"}],value="words"),
                                    html.Br(),
                                            dbc.Button("Analyze", id="chain_button",color="primary",block=True),
                                            dbc.Progress(id="progress",value=0)
                                        ]),
                                    html.Div(id="alert",children=[])
                                                                        #html.H6("Boundary Condition:"),
                                    #dcc.RadioItems(id='condition',options=[{"label":"no","value":"no"},{"label":"periodic","value":"periodic"},{"label":"ordinary","value":"ordinary"}],value="words"),
                                ]


                                            ),
                                dbc.CardHeader("Characteristics"),
                                dbc.CardBody(
                                    [
                                        html.Div(id="lenght",children=["Lenght: ",]),
                                        html.Div(id="vocabulary",children=["Vocabulary: ",]),
                                        html.Div(id="chain_time",children=["Time: ",]),
                                        dbc.Spinner(dbc.Button("Save data",id="save",color="success",size="md")),
                                        html.Div(id="temp_seve",
                                             children=[]
                                            )


                                    ]
                                            )

                            ],color="light",style={"margin-left":"10px","margin-top":"10px",}
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
                                                 style={"display":"none"},
                                                 children=[dbc.Spinner(dt.DataTable(
                                                    id="table",
                                                columns=[{"name":i,"id":i}for i in ['rank',"ngram","ƒ","R","a","b","goodness"]],
                                                style_data={'whiteSpace': 'auto','height': 'auto'},
                                                editable=False,
                                                filter_action="native",
                                                sort_action="native",
                                                page_size=30,
                                                fixed_rows={'headers': True},
                                                style_cell={'whiteSpace': 'normal',
                                                             'height': 'auto',
                                                            'textAlign': 'right',
                                                             "fontSize":15,
                                                            "font-family":"sans-serif"},#'minWidth': 40, 'width': 95, 'maxWidth': 95},
                                            style_table={'height': 420, 'overflowY': 'auto',"overflowX":"none"}
                                                                                    ))]),
                                        html.Div(id="box_chain",
                                                 style={"display":"none"},
                                                 children=[dbc.Spinner(dcc.Graph(id="chain"))])


                                     ]
                                             )
                                ],style={"margin-right":"10px","margin-top":"10px"}),
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            dbc.Tabs(
                                                [
                                                    dbc.Tab(label="distribution",tab_id="tab1"),
                                                    dbc.Tab(label="flunctuacion",tab_id="tab2"),
                                                    dbc.Tab(label="alpha/R",tab_id="tab3")
                                                ],
                                                id='card-tabs',
                                                card=True,
                                                active_tab="tab1"
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
                                            dcc.Graph(id="graphs")

                                        ])

                                    ],style={"margin-right":"10px"}
                                )
                            ],
                        width={"size":9}
                            ),
                                        ]
                        )])
from dash.dependencies import Input,Output,State
app.layout=layout1
df=None
import plotly.express as px
from sklearn.metrics import r2_score
import networkx as nx

@app.callback([Output("w","value"),
               Output("wh","value"),
              # Output("we","value"),
               Output("wm","value"),
               Output("lenght","children"),],
               [Input("corpus","value")],Input("split","value"))
def calc_window(corpus,split):
    if corpus is None:
        return dash.no_update,dash.no_update,dash.no_update,dash.no_update
    global L,data
    
    with open("corpus/"+corpus) as f:
        file=f.read()
        file=unicodedata.normalize("NFKD",file)
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
        data.replace(" ","space")
        data=data.split()
        


    
    L=len(data)
    wm=int(L/10)
    w=int(wm/10)
    return [w,w,wm,["Lenght: ",L]]
@app.callback([Output("table","data"),Output("chain","figure"),Output("box_tab","style"),Output("box_chain","style"),Output("alert","children"),Output("vocabulary","children"),Output("chain_time","children")],
              [Input("chain_button","n_clicks"),
               Input("progress","value"),
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
               ])
def update_table(n,progress,dataframe,corpus,n_size,split,condition,f_min,w,wh,we,wm):
    
    
    global data,L,V,model,ngram,df
    if dataframe=="markov_chain":

        if n is None:
            
            return dash.no_update,dash.no_update,{"display":"none"},{"display":'inline'},dash.no_update,dash.no_update,dash.no_update

        #add alert corpus if not selected
        if corpus is None:
            return dash.no_updata,dash.no_update,{"display":"none"},{"display":"inline"},dbc.Alert("Please choose corpus",color="danger",duration=2000,dismissable=False),dash.no_update,dash.no_update

        
       ## make markov chain graph ###
        g=nx.MultiGraph()
        temp={}

        for ngram in df['ngram']:
            if n_size>1:
                ngram=tuple(ngram.split())
            
            g.add_node(ngram)
            temp[ngram[0]]=ngram
  
        for node in g.nodes():
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
                node_text.append(" ".join(adjacencies[0])+'<br>connections: '+str(len(adjacencies[1])))
                continue
            node_text.append("".join(adjacencies[0])+'<br>connections: '+str(len(adjacencies[1])))

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


        ###  MAKE MARKOV CHAIN ####
        start=time()
        make_markov_chain(data,order=n_size,split=split)
        df=make_dataframe(model,L,f_min)
        for index,ngram in enumerate(df['ngram']):
            print(str(index)+" of "+str(len(df['ngram'])),end="\r")
            model[ngram].dt=calculate_distance(np.array(model[ngram].pos,dtype=np.uint8),L,condition)
        windows=list(range(w,wm,we))
        fa(np.zeros(5),(1,3,4))
        def func(wind):
            model[ngram].counts[wind],model[ngram].fa[wind]=fa(model[ngram].bool,(wind,wh,L))
        for index,ngram in enumerate(df['ngram']):
            print(str(index)+" of "+str(len(df['ngram'])),end="\r")
            with ThreadPoolExecutor() as e:
                e.map(func,windows)
        #calculate_fa(df,model,w,wh,we,wm,L,condition)
        ###
        temp_b=[]

        temp_R=[]
        temp_error=[]
        temp_ngram=[]
        temp_a=[]

        for ngram in df['ngram']:
            model[ngram].temp_fa=[]
            c,cov=curve_fit(fit,[*model[ngram].fa.keys()],[*model[ngram].fa.values()],method='lm',maxfev=5000)
            model[ngram].a=c[0]
            model[ngram].b=c[1]
            for w in model[ngram].fa.keys():
                model[ngram].temp_fa.append(fit(w,model[ngram].a,model[ngram].b))
            temp_error.append(round(r2_score([*model[ngram].fa.values()],model[ngram].temp_fa),5))
            temp_b.append(round(c[1],7))
            temp_a.append(round(c[0],7))

            
            if ngram.__class__ is tuple:
                temp_ngram.append("  ".join(ngram))
            #print(np.array(model[ngram].dt))
            temp_R.append(round(R(np.array(model[ngram].dt)),7))

        if n_size>1:
            df["ngram"]=temp_ngram
        df['R']=temp_R
        df['b']=temp_b 
        df['a']=temp_a
        df['goodness']=temp_error

        df=df.sort_values(by="ƒ",ascending=False)
        df['rank']=range(1,len(temp_R)+1)
        df=df.set_index(pd.Index(np.arange(len(df))))
        print(df)
        
        #table.data=df.to_dict("records")
        return [df.to_dict("record"),dash.no_update,{"display":"inline"},{"display":"none"},dash.no_update,["Vocabulary: ",V],["Time: ",round(time()-start,6)]]

@app.callback(Output("graphs","figure"),
              [Input("card-tabs","active_tab"),
                Input("table","active_cell"),
               Input("table","derived_virtual_selected_rows"),
               Input("table","derived_virtual_indices"),
               Input("scale","value")
               ],[State("n_size","value")])
def tab_content(active_tab,active_cell,row_ids,ids,scale,n):
    global model,df,L
    if df is None:
        return dash.no_update
    if ids is None:
        return dash.no_update

    df=df.reindex(pd.Index(ids))
    fig=go.Figure()
    fig.update_layout(
        margin=dict(l=0,r=0,t=0,b=0)
    )
    if active_tab=="tab1":
        if active_cell:
            ngram=''
            if n>1:
                print(df['ngram'][0])
                ngram=tuple(df['ngram'][ids[active_cell['row']]].split())
            else:
                ngram=df['ngram'][ids[active_cell['row']]]
            fig.add_trace(go.Scatter(x=np.array(range(L)),y=model[ngram].bool))
            fig.update_xaxes(type=scale)
            return fig
        else:
            return fig
        return fig
    if active_tab=="tab2":
        if active_cell:
            if n>1:
                ngram=tuple(df['ngram'][ids[active_cell['row']]].split())
            else:
                ngram=df['ngram'][ids[active_cell['row']]]
            fig.add_trace(
                    go.Scatter(x=[*model[ngram].fa.keys()],
                                     y=[*model[ngram].fa.values()],
                                     mode='markers',
                                    name="∆F"))
            fig.add_trace(go.Scatter(
                                x=[*model[ngram].fa.keys()],
                                y=model[ngram].temp_fa,
                                name="fit"))
            fig.update_xaxes(type=scale)
            fig.update_yaxes(type=scale)
            return fig
        else:
            return fig
    else:
        hover_data=[]
        if active_cell:
            if n>1:
                ngram=tuple(df['ngram'][ids[active_cell['row']]].split())
            else:
                ngram=df['ngram'][ids[active_cell['row']]]
            
            for data in df['ngram']:
                hover_data.append("".join(data))
            fig.add_trace(go.Scatter(x=df["R"],y=df["b"],mode="markers",text=hover_data))
            fig.add_trace(go.Scatter(x=[df['R'][active_cell['row']]],
                                     y=[df["b"][active_cell['row']]],
                                     mode="markers",
                                     text=' '.join(ngram),
                                     marker=dict(
                                         size=20,
                                         color="red"
                                     ))) 
            fig.update_layout(showlegend=False)
            fig.update_yaxes(type=scale)
            fig.update_xaxes(type=scale)
            return fig
        else:
            
            for data in df["ngram"]:
                hover_data.append("".join(data))
            fig.add_trace(go.Scatter(x=df["R"],y=df["b"],mode="markers",text=hover_data))
            fig.update_yaxes(type=scale)
            fig.update_xaxes(type=scale)
        return fig        

        return dash.no_update
@app.callback([Output("temp_seve","children")],
              [Input("save","n_clicks")])
def save(n):
    if n is None :
        return dash.no_update
    else:
        print("here")
        global df
        print(df)
        writer=pd.ExcelWriter("output.xlsx")
        df.to_excel(writer)
        writer.save()
        print("done")
    return dash.no_update


import webbrowser
if __name__=="__main__":
    webbrowser.open("http://127.0.0.1:8050/")

    app.run_server(debug=True)

    #webbrowser.open("http://127.0.0.1:8050/")
    #main()





"""
                                        html.Div(children=[dbc.Spinner(dt.DataTable(id='table',
                                                columns=[{"name":i,"id":i}for i in ['rank',"ngram","ƒ","R","a","b","goodness"]],
                                                style_data={'whiteSpace': 'auto','height': 'auto'},
                                                editable=False,
                                                filter_action="native",
                                                sort_action="native",
                                                page_size=10,
                                                fixed_rows={'headers': True},
                                                style_cell={'whiteSpace': 'normal',
                                                             'height': 'auto',
                                                            'textAlign': 'right',
                                                             "fontSize":15,
                                                            "font-family":"sans-serif"},#'minWidth': 40, 'width': 95, 'maxWidth': 95},
                                            style_table={'height': 450, 'overflowY': 'auto',"overflowX":"none"}
                                                  ))],  style={"display","none"}),
                                                html.Div(children=[
                                                    dbc.Spinner(
                                                    dcc.Graph(id='chain'))],
                                                                      id="box_chain",
                                                                      style={"display":'none'})
"""






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
"""
#print("fa time:",time()-start)
#print(df)
#L,V=0,0
##wh=0
#model=0
#ngram=0
#@profile
#def main():
#    global L,V,wh,model,ngram
#    with open("corpus/lotr_en.txt") as f:
#        file=f.read()
#    data=remove_punctuation(file)
#    start=time()
#    fmin=4
#    order=1
#    split="word"
#    option="obc"
#    make_markov_chain(data.split(),order=order,split=split)
#
#    #model
#    df=make_dataframe(model,fmin)
#    print("chain time:",time()-start)
#    #print(df)
#
#    ### CALCULATE FA ###
#
#    print("order:",order)
#    print("split by:",split)
#    print("L:",L)
#    print("V: ",V)
#    print("fmin:",fmin)
#    print("valid V:",len(df['ngram']))
#    print()
#    wmax=int(L/20)
#    w=int(wmax/10)
#    we=int(wmax/10)
#    wh=w
#    print("w:",w)
#    print("wmax:",wmax)
#    print("we:",we)
#    print("wh:",wh)
#    print("option:",option)
#    #model['entropy'].bool
#
#
#
#    start=time()
#    temp_w=0
##    g()
# #   input()
#
#    calculate_fa(df,model,w,wmax,we,wh,L,option)
#    temp_b=[]
#    temp_fi=[]
#    temp_R=[]
##    print(df)
#    print()
#    print("fa time:",time()-start)
#    #for ngram in model:
#        #print(model[ngram].fa)
#
#
#
#    for ngram in df['ngram']:
#        c,cov=curve_fit(fit,[*model[ngram].fa.keys()],[*model[ngram].fa.values()],maxfev=5000)
#        model[ngram].a=c[0]
#        model[ngram].b=c[1]
#        temp_b.append(c[1])
#        temp_fi.append(model[ngram].F_i/L)
#        temp_R.append(R(np.array(model[ngram].pos)))
#    df['R']=temp_R
#    df['f_i']=temp_fi
#    df['alpha']=temp_b
#    print(df)
#
#
#
#    pass
#





#@app.callback([Output("table","data")],
#              [Input("wh","value")])
#def add_fa_analyze(wh):
#    return dash.no_update
    #return df,dash.no_update
#app.layout=html.Div(style={"backgroundColor":colors["background"]},
#                    children=[html.Div(style={"backgroundColor":colors["text"],
#                                                "widht":"20%",
#                                              "float":"left",
#                                              "text-align":"center",
#                                              "padding":"1%"},
#                              children=[
#                                        html.Div(children=[html.Label("Choose corpus"),
#                                        dcc.Dropdown(
#                                            id="corpus",
#                                            options=[{"label":i,"value":i} for i in corpus],
#                                        )],style={"width":"90%",
#                                                  "margin-right":"5%",
#                                                  "margin-left":"5%"}),
#                                        html.Div(children=[
#                                        html.Label("size of ngram"),
#                                        dcc.Slider(
#                                            id="n-size",
#                                            min=1,
#                                            max=7,
#                                            marks={i:"{}".format(i)for i in range(1,8)},
#                                            value=1)],style={"width":"80%",
#                                                             "margin-left":"10%",
#                                                             "margin-right":"10%"}),
#                                        html.Div(children=[
#                                        html.Label("split by"),
#                                        dcc.Dropdown(
#                                            id="split",
#                                            options=[{"label":i,"value":i} for i in ["word","symbol"]],
#                                            value="word")],style={"width":"60%",
#                                                                  "margin-left":"20%",
#                                                                  "margin-right":"20%"}),
#                                        html.Div(children=[
#                                         html.Label("Boundary Condition"),
#                                        dcc.Dropdown(
#                                            id="options",
#                                            options=[{"label":i,"value":i} for i in ["no","periodic","ordinary"]],
#                                            value="no")],style={"width":"60%",
#                                                                "margin-left":"20%",
#                                                                "margin-right":"20%"}),
#                                        html.Div(children=[
#                                        html.Label("freauency of ngram"),
#                                        dcc.Input(
#                                            id="fmin",
#                                            style={"padding":"5%","width":"50%"},
#                                            placeholder="filter",
#                                            type='number',
#                                            debounce=True,
#                                            value="")],style={"width":'60%',
#                                                            "margin-left":"19%",
#                                                             "margin-right":"19%"})
#
#                                        ]),
#                              html.Div(style={"float":"right","width":"75%"},
#                                       children=[dt.DataTable(
#                                       id='table',
#                                           columns=[{"name":i,"id":i}for i in ["ngram","F_i","f_i","R","alpha"]],
#                                       data=[])])])
#from dash.dependencies import Input,Output,State
#@app.callback(Output("table","data"),
#              [Input("corpus","value"),
#               Input("n-size","value"),
#               Input("split","value"),
#               Input("options","value"),
#               Input("fmin","value")])
#def update_table(corpus,n_size,split,options,fmin):
#    print(corpus)
#    print(n_size)
#    print(split)
#    print(options)
#    print(fmin)
#    if corpus is None:
#        pass
#    else:
#
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
 #       pass



