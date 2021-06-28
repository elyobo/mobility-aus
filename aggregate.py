###############################################################################
###############################################################################


import os
import pickle
from collections.abc import Sequence
from itertools import product

import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame as gdf
import mercantile
import shapely

import aliases
import load
from utils import update_progressbar


def quadkeys_to_poly(quadkeys):
    quadkeys = sorted(set(quadkeys))
    tiles = [mercantile.quadkey_to_tile(qk) for qk in quadkeys]
    tiles = mercantile.simplify(tiles)
    quadkeys = [mercantile.quadkey(t) for t in tiles]
    polys = quadkeys_to_polys(quadkeys)
    poly = shapely.ops.unary_union(polys)
    return poly

def make_quadkeypairs(fbdata):
    out = fbdata.reset_index()[['quadkey', 'end_key']]
    out.drop_duplicates()
    return out

def make_quadfrm(quadkeys):
    quadkeys = sorted(set(quadkeys))
    quadpolys = quadkeys_to_polys(quadkeys)
    quadFrm = gdf(geometry = quadpolys, index = quadkeys)
    quadFrm.index.name = 'quadkey'
    return quadFrm


def make_intersection_weights(fromFrm, toFrm):
    joined = gpd.tools.sjoin(fromFrm, toFrm, 'left', 'intersects')
    joined = joined.fillna(0)
    groupby = joined['index_right'].groupby(joined.index)
    def agg_func(s):
        nonlocal fromFrm
        nonlocal toFrm
        toIndices = [int(val) for val in list(set(s))]
        if len(toIndices) == 1:
            return [(toIndices[0], 1.)]
        toPolys = [toFrm.loc[i]['geometry'] for i in toIndices]
        fromIndex = s.index[0]
        fromPoly = fromFrm.loc[fromIndex]['geometry']
        weights = [fromPoly.intersection(p).area for p in toPolys]
        weights = [w / sum(weights) for w in weights]
        return list(zip(toIndices, weights))
    weights = groupby.aggregate(agg_func)
    weights = dict(zip(weights.index, list(weights)))
    return weights

def split_iterate(frm):
    print("Splitting frame by possible journey...")
    maxi = len(frm)
    for i, row in frm.iterrows():
        journeys = row['possible_journeys']
        for start, stop, weight in journeys:
            yield *row, start, stop, weight
    print("Frame split.")
def split_journeys(frm):
    frm = frm.reset_index()
    iterator = split_iterate(frm)
    columns = zip(*iterator)
    newcolumnnames = [*frm.columns, 'start', 'stop', 'weight']
    frm = pd.DataFrame(dict(zip(newcolumnnames, columns)))
    frm = frm.drop(['quadkey', 'end_key', 'possible_journeys'], axis = 1)
    frm = frm.set_index(['datetime', 'start', 'stop'])
    return frm


class SpatialAggregator:

    __slots__ = ('aggtype', 'diskpath')

    def __init__(self, aggtype = 'lga'):
        self.aggtype = aggtype
        self.diskpath = os.path.join(aliases.cachedir, f"quadweights_{aggtype}.pkl")

    @property
    def weights(self):
        try:
            with open(self.diskpath, mode = 'rb') as file:
                return pickle.loads(file.read())
        except FileNotFoundError:
            return dict()
    def store(self, tostore):
        with open(self.diskpath, mode = 'wb') as file:
            file.write(pickle.dumps(tostore))
    @property
    def tofrm(self):
        return load.load_generic(self.aggtype)

    def get_quadkey_weights(self, quadkeys):
        print("Getting quadkey weights...")
        quadkeys = sorted(set(quadkeys))
        weights = self.weights
        tocalc = [key for key in quadkeys if not key in weights]
        if not tocalc:
            print("Quadkey weights retrieved.")
            return weights
        quadfrm = make_quadfrm(tocalc)
        newweights = make_intersection_weights(quadfrm, self.tofrm)
        weights.update(newweights)
        self.store(weights)
        print("Quadkey weights calculated.")
        return {key: weights[key] for key in quadkeys}

    def __getitem__(self, arg):
        if isinstance(arg, str):
            return self[[arg,]][arg]
        if isinstance(arg, (Sequence, set)):
            return self.get_quadkey_weights(arg)
        if isinstance(arg, pd.DataFrame):
            return self[set((
                *arg.index.get_level_values('quadkey'),
                *arg.index.get_level_values('end_key'),
                ))]

    @staticmethod
    def _poss_journey_groupfunc(x):
        startWeights, endWeights = x[['start_weights', 'end_weights']].values[0]
        possibleJourneys = list(product(startWeights, endWeights))
        outRows = []
        for pair in possibleJourneys:
            (start, startWeight), (end, endWeight) = pair
            outRow = [int(start), int(end), startWeight * endWeight]
            outRows.append(outRow)
        return outRows

    def add_possible_journeys(self, frm):
        print(f"Adding possible journeys...")
        weights = self[frm]
        indexnames = frm.index.names
        frm = frm.reset_index()
        frm['start_weights'] = frm.reset_index()['quadkey'].apply(
            lambda x: weights[x]
            )
        frm['end_weights'] = frm.reset_index()['end_key'].apply(
            lambda x: weights[x]
            )
        groupby = frm.groupby(['quadkey', 'end_key'])
        groupby = groupby[['start_weights', 'end_weights']]
        frm = frm.reset_index().set_index(['quadkey', 'end_key'])
        frm['possible_journeys'] = groupby.apply(self._poss_journey_groupfunc)
        frm = frm.reset_index().set_index(indexnames)
        frm = frm.drop({'start_weights', 'end_weights', 'index'}, axis = 1)
        print(f"Added possible journeys.")
        return frm

    def aggregate(self, frm):
        print(f"Aggregating to {self.aggtype}...")
        frm = self.add_possible_journeys(frm)
        frm = split_journeys(frm)
        print("Aggregated.")
        return frm

    def __call__(self, frm):
        return self.aggregate(frm)


def combine_by_date(frm):
    print("Combining dates...")
    indexnames = frm.index.names
    frm = frm.reset_index()
    frm['date'] = frm['datetime'].apply(lambda x: x.date)
    frm['time'] = frm['datetime'].apply(lambda x: x.time)
    frm = frm.drop('datetime', axis = 1)
    frm = frm.set_index(['date', *indexnames[1:]])
    print("Dates combined.")
    return frm


def aggregate(frm, aggtype = 'lga'):
    return combine_by_date(SpatialAggregator(aggtype)(frm))


###############################################################################
###############################################################################
