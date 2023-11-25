# osm_net_to_matsim
Converts several types of OSM network (road, tram, rail, metro) into MATSim XML format. This tool preserves the original geometry

## Linux, macOS
### Install
Open a terminal in the selected folder or `cd` there, then type:

```
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```
### Run
Considering you're running the activated venv: 

`python osm_net_to_matsim.py -p "<path to OSM protocol buffer (.osm.pbf)>" -c "<output CRS of shapefiles, like epsg:5514, optional>" -o "<path to shapefile with region outer border, optional>" -d "<path to shapefile where high detailed roads should be kept, optional>" -n "<output path to MATSim net XML file>" -e "<output path to net edges shapefile, optional>" -N "<output path to net nodes shapefile, optional>"`

## Windows
### Install
Open a terminal in the selected folder or `cd` there, then type:
```
python -m venv .\venv
.\venv\Scripts\activate.bat
pip install -r requirements.txt
```
Might fail on `geopandas` install. If it does, refer to `geopandas install windows` instructions on the internet

### Run
Considering you're running the activated venv: 

`python osm_net_to_matsim.py -p "<path to OSM protocol buffer (.osm.pbf)>" -c "<output CRS of shapefiles, like epsg:5514, optional>" -o "<path to shapefile with region outer border, optional>" -d "<path to shapefile where high detailed roads should be kept, optional>" -n "<output path to MATSim net XML file>" -e "<output path to net edges shapefile, optional>" -N "<output path to net nodes shapefile, optional>"`

