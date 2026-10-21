from pathlib import Path
import urllib.request, pandas as pd, folium
ROOT=Path(__file__).resolve().parents[1]
csv=ROOT/"data"/"dataset_part_2.csv"; csv.parent.mkdir(exist_ok=True)
if not csv.exists():
 urllib.request.urlretrieve("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv",csv)
df=pd.read_csv(csv)
m=folium.Map(location=[28.5,-95],zoom_start=4,tiles="OpenStreetMap")
for _,r in df.dropna(subset=["Latitude","Longitude"]).iterrows():
 outcome="Successful" if int(r["Class"])==1 else "Unsuccessful"
 folium.CircleMarker([r["Latitude"],r["Longitude"]],radius=5,
   popup=f'Flight {r["FlightNumber"]} | {r["LaunchSite"]} | {outcome}',
   tooltip=f'{r["Date"]}: {outcome}',
   color=("green" if outcome=="Successful" else "red"),fill=True,fill_opacity=.7).add_to(m)
out=ROOT/"outputs"; out.mkdir(exist_ok=True)
m.save(out/"launch_map.html")
print("Saved outputs/launch_map.html")
