import pandas as pd
import numpy as np
import time as t
import copy
import math


def process(x1, x2):
    def asignacion(df, centroids):
        colmap = {1: 'r', 2: 'g', 3: 'b'}
        for i in centroids.keys():
            # sqrt((x1 - c1)^2 - (x2 - c2)^2)
            df['distance_from_{}'.format(i)] = (
                np.sqrt(
                    (df['x1'] - centroids[i][0]) ** 2
                    + (df['x2'] - centroids[i][1]) ** 2
                )
            )
        centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
        df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
        df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
        df['color'] = df['closest'].map(lambda x: colmap[x])
        return df
    def update(k):
        for i in centroids.keys():
            centroids[i][0] = np.mean(df[df['closest'] == i]['x1'])
            centroids[i][1] = np.mean(df[df['closest'] == i]['x2'])
        return k

    t0 = t.time()
    colmap = {1: 'r', 2: 'g', 3: 'b'}

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2
    })

    np.random.seed(200)

    k = 3

    centroids = {
        i+1: [np.random.randint(0, 296), np.random.randint(0, 560)]
        for i in range(k)
    }


    df = asignacion(df, centroids)

    old_centroids = copy.deepcopy(centroids)


    centroids = update(centroids)
    df = asignacion(df, centroids)
    # print("time: ", t.time() - t0)
    # print(centroids)
    res = []
    for i in centroids.keys():
        if not math.isnan(centroids[i][0]):
            res.append([int(centroids[i][0]), int(centroids[i][1])])
    return res


