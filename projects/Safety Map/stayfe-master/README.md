Stayfe
=======
Created a unique method of crime data crawling from public news media sources and used natural language processing libraries to parse free texts, keyword filtering to classify types of crime along with statistical methods to produce "safest path" suggestion for pedestrians. We transformed news article data into coordinates and mapped onto Google Maps, and eventually created a safety-prevention wep app.

To run the flask server for computing safest path between two points:

Requirements
--------------

- python 3
- flask
- googlemaps
- polyline
- numpy
- pandas
- scikit-learn


Endpoints
-----------------------

Run the server:

```
python app.py
```

Get crime data between two points:
```
GET /crimes/:src&:dst
```

```
e.g. http://host:port/crimes?src=-122.445,37.74&dst=-122.42,37.715
```

Find the safest path between two points:
```
GET /path/:src&:dst
```

```
e.g. http://host:port/path?src="Hillegaas Avenue, Berkeley, CA"&dst="Soda Hall, Berkeley, CA"
```

