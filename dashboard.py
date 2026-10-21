from pathlib import Path
import urllib.request
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

ROOT=Path(__file__).resolve().parents[1]
csv=ROOT/"data"/"dataset_part_2.csv"
csv.parent.mkdir(exist_ok=True)
if not csv.exists():
    urllib.request.urlretrieve("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv",csv)
df=pd.read_csv(csv)
app=Dash(__name__)
sites=sorted(df["LaunchSite"].dropna().unique())
app.layout=html.Div([
 html.H1("SpaceX Falcon 9 Landing Analytics"),
 html.Label("Launch site"),
 dcc.Dropdown(options=[{"label":"All sites","value":"ALL"}]+[{"label":s,"value":s} for s in sites],value="ALL",id="site"),
 html.Label("Payload mass range (kg)"),
 dcc.RangeSlider(id="payload",min=0,max=int(df["PayloadMass"].max(skipna=True)),step=500,
                 value=[0,int(df["PayloadMass"].max(skipna=True))],marks=None,tooltip={"placement":"bottom","always_visible":True}),
 dcc.Graph(id="success-pie"), dcc.Graph(id="payload-scatter")
],style={"fontFamily":"Arial","maxWidth":"1100px","margin":"auto"})
@app.callback(Output("success-pie","figure"),Output("payload-scatter","figure"),Input("site","value"),Input("payload","value"))
def update(site,payload):
 d=df[df["PayloadMass"].fillna(0).between(payload[0],payload[1])]
 if site!="ALL": d=d[d["LaunchSite"]==site]
 pie=d["Class"].map({0:"Unsuccessful",1:"Successful"}).value_counts().rename_axis("Outcome").reset_index(name="Launches")
 fig1=px.pie(pie,names="Outcome",values="Launches",title="Landing outcome distribution")
 fig2=px.scatter(d,x="PayloadMass",y="FlightNumber",color=d["Class"].map({0:"Unsuccessful",1:"Successful"}),
                 hover_data=["LaunchSite","Orbit"],title="Payload mass vs flight number",labels={"color":"Outcome"})
 return fig1,fig2
if __name__=="__main__": app.run(debug=True)
