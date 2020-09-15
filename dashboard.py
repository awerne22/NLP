import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt

app=dash.Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
app.layout=html.Div(style={'backgroundColor':colors["background"]},
                    children=[html.Label("tabl",style={"color":colors["text"]}),
                              dt.DataTable()])

if __name__ =="__main__":
    app.run_server(debug=True)


