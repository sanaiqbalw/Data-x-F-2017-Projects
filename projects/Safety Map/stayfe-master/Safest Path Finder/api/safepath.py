import numpy as np
import pandas as pd
from skimage import measure


# load the data onto memory
df = pd.read_csv("../data/light_full_classified.csv")


def get_bbox(A,B):
    x0 = np.amin([A[0], B[0]])
    x1 = np.amax([A[0], B[0]])
    y0 = np.amin([A[1], B[1]])
    y1 = np.amax([A[1], B[1]])
    dx = x1-x0
    dy = y1-y0
    return (x0-2*dx, y0-2*dy, x0+2*dx, y0+2*dy)


def get_time_diff(t1,t2):
    try:
        dtime = (t1 - t2).dt.total_seconds() 
    except:
        dtime = -1
    return dtime


def get_crime_data(bbox):
    query = df.loc[(df['Lon']>bbox[0]) & (df['Lat']>bbox[1]) & (df['Lon']<bbox[2]) & (df['Lat']<bbox[3])]
    DT = pd.to_datetime(query['Time'])
    now = pd.to_datetime('now')
    query = query.assign(secsago = get_time_diff(now, DT))
    
    def get_width(cat):
    	if "gun" or "bomb" in cat:
        	return 0.01
    	else:
        	return 0.001
    
    crimes = np.asarray(query[['Lon', 'Lat']])
    coords = np.expand_dims(crimes.T, axis=1)
    amps = np.exp(-query['secsago']/1e8).values.reshape((1, coords.shape[-1]))
    widths = np.array([get_width(cat) for cat in query['Weapon']]).reshape(1,1, coords.shape[-1])

    return crimes, coords, amps, widths


def get_weights(arr, coords, widths, amps, midpoint=False):
    """
    arr: array of points to evaluate at, shape (2, None, 1)
    coords: array of points of crimes, shape (2, 1, None)
    """
    if midpoint:
        arr = (arr[...,1:,:]+arr[...,:-1,:])/2
    return np.sqrt(np.sum(amps*np.exp(-np.mean(((arr-coords)/widths)**2, axis=0)), axis=-1))


def get_deriv(pt, coords, widths, amps):
    widths = widths[np.newaxis,...]
    amps = amps[np.newaxis, ...]
    num = np.sum(-(pt-coords)/widths**2*amps*
                  np.exp(-np.mean(((pt-coords)/widths)**2, axis=0)), axis=-1)
    den = np.sqrt(np.sum(amps*np.exp(-np.mean(((pt-coords)/widths)**2, axis=0)), axis=-1))
    return num/den


def get_weighted_distance(arr, coords, widths, amps):
    weights = get_weights(arr, coords, widths, amps, midpoint=True)
    dists = (arr[...,1:,:]-arr[...,:-1,:]).squeeze()
    dists = np.hypot(dists[0], dists[1])
    return np.dot(dists, weights)


def get_curve(waypoints):
    K = len(waypoints.keys())

    if K == 2:
        X = np.linspace(waypoints[0][0], waypoints[1][0], 10).reshape((10,1))
        Y = np.linspace(waypoints[0][1], waypoints[1][1], 10).reshape((10,1))
        arr = np.stack([X,Y], axis=0)
    else:
        U = []; CV = []
        
        for u in sorted(waypoints):
            U.append(u)
            CV.append([waypoints[u][0],waypoints[u][1]])

        CV = np.asarray(CV).T
        x_params = np.polyfit(U, CV[0], K-1)
        y_params = np.polyfit(U, CV[1], K-1)
        y = np.poly1d(y_params)
        x = np.poly1d(x_params)
        t = np.linspace(0,1,10)
        arr = np.stack([x(t),y(t)], axis=0).reshape((2,10,1))
    
    return arr


def get_waypoints(A,B):
    bbox = get_bbox(A,B)
    crimes, coords, amps, widths = get_crime_data(bbox)
    pts = 0
    safe = False
    threshold = 0.02
    waypoints = {0:A, 1:B}
    it = 0

    while not safe:
        arr = get_curve(waypoints)
        weights = get_weights(arr, coords, widths, amps)
        danger = weights > threshold

        if not any(danger):
            safe = True
            break
        
        label = measure.label(danger)
        N = np.amax(label)
        
        for n in range(1, N+1):
            s = int(np.mean([i for i, x in enumerate(label) if x == n]))
            start = arr[:,s,0] # shape (2,)
            dU = get_deriv(start.reshape((2,1)), coords.squeeze(), widths.squeeze(), amps.squeeze())
            start -= dU*5e-7
            c = 0

            while (c < 20 and get_weights(start.reshape((2,1,1)), coords, widths, amps) > threshold * 0.5):
                c += 1
                start -= dU*5e-8
                dU = get_deriv(start.reshape((2,1)), coords.squeeze(), widths.squeeze(), amps.squeeze())
            
            u = float(s)/10
            waypoints[u] = start
        
        it += 1
        
        if it > 4:
            break

    return waypoints, crimes
    
    
if __name__ == "__main__":
    A = np.array([-122.2558202,37.8598449])
    B = np.array([-122.2587865,37.8755939])
    print(get_waypoints(A,B))
