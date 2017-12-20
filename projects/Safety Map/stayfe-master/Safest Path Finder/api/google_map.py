import googlemaps
import polyline
from datetime import datetime

from safepath import get_waypoints

gmaps = googlemaps.Client(key='AIzaSyDCXWvoPjy1rtQVJ5AqQBC2y8tGQQwOnas')


def decode_polyline(dr):
    path = polyline.decode(dr['overview_polyline']['points'])
    return path

def geocode(addr):
    res = gmaps.geocode(addr)[0]['geometry']['location']
    return (res['lat'], res['lng'])

# waypoints: a single location, or a list of locations
# optimize_waypoints: let google reorder waypoints
def googleDirection(src, dst, wypoints=None):
    now = datetime.now()
    directions = gmaps.directions(src,
                                  dst,
                                  mode='walking',
                                  waypoints=wypoints,
                                  departure_time=now)
    
    return directions

def compute_path(src, dst):
    src_addr = src
    dst_addr = dst

    if isinstance(src_addr, str):
        src_addr = geocode(src_addr)
    if isinstance(dst_addr, str):
        dst_addr = geocode(dst_addr)

    ss = (src_addr[1], src_addr[0])
    dd = (dst_addr[1], dst_addr[0])
    ways, crimes = get_waypoints(ss, dd)
    
    out = []
    sort_keys = sorted(ways.keys())
    for k in sort_keys:
        out.append(tuple(reversed(ways[k])))
    out = out[1:-1]

    path = googleDirection(src, dst)
    p = path[0]
    shortest_path = decode_polyline(p)

    waypoints = googleDirection(src, dst, out)
    w = waypoints[0]
    safest_path = decode_polyline(w)

    return shortest_path, safest_path, crimes

if __name__ == '__main__':
    print(compute_path("Hillegaas Avenue, Berkeley, CA", "Soda Hall, Berkeley, CA"))
