###############################################################################
''''''
###############################################################################


import os as os
import sys
from glob import glob as glob
from datetime import datetime as datetime, timezone as timezone
from collections.abc import Sequence
from functools import partial, lru_cache
import requests, zipfile, io
import pickle

import pandas as pd
from dask import dataframe as daskdf
import numpy as np
import shapely
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import itertools

df = pd.DataFrame
import geopandas as gpd
gdf = gpd.GeoDataFrame

from utils import update_progressbar
import utils
import aliases

REPOPATH = os.path.dirname(__file__)


STATENAMES = {
    'vic': 'Victoria',
    'nsw': 'New South Wales',
    'qld': 'Queensland',
    'sa': 'South Australia',
    'wa': 'Western Australia',
    'tas': 'Tasmania',
    'nt': 'Northern Territory',
    'act': 'Australian Capital Territory',
    'oth': 'Other Territories',
    }

GCCNAMES = {
    'mel': 'Greater Melbourne',
    'syd': 'Greater Sydney'
    }

GCCSTATES = {
    'mel': 'vic',
    'syd': 'nsw',
    }

def get_state(region):
    return (
        STATENAMES[region if region in STATENAMES else GCCSTATES[region]]
        )


def process_datetime(x):
    x = x.replace(':', '').replace('_', ' ')
    stripped = datetime.strptime(x, '%Y-%m-%d %H%M')
    adjusted = stripped.astimezone(timezone.utc)
    return adjusted

def datetime_str_from_datafilename(fname):
    name, ext = fname.split('.')
    name = name[name.index('_')+1:]
    return name

def datafilename_to_datetime(fname):
    name = datetime_str_from_datafilename(fname)
    return process_datetime(name)


class FBDataset:

    __slots__ = (
        'reg', 'fbid', 'datadir', 'timezone', 'region',
        'prepath', '_prefrm', 'h5path', '_daskfrm',
        )

    FBIDS = {
        'mel': '786740296523925',
        'vic': '1391268455227059',
        'syd': '1527157520300850',
        'nsw': '2622370339962564',
        }

    RENAMEDICT = {
        'date_time': 'datetime',
        'start_quadkey': 'quadkey',
        'start_quad': 'quadkey',
        'end_quadkey': 'end_key',
        'end_quad': 'end_key',
        'length_km': 'km',
        'n_crisis': 'n'
        }

    KEEPKEYS = ['datetime', 'quadkey', 'end_key', 'km', 'n']

    PROCFUNCS = {
        'km': float,
        'n': int,
        'quadkey': str,
        'end_key': str,
        }

    TZS = {
        'vic': 'Australia/Melbourne',
        'mel': 'Australia/Melbourne',
        'nsw': 'Australia/Sydney',
        'syd': 'Australia/Sydney',
        }

    def __init__(self, region):
        self.region = region
        fbid = self.fbid = self.FBIDS[region]
        repopath = aliases.repodir
        datadir = self.datadir = os.path.join(aliases.datadir, fbid)
        self.prepath = os.path.join(aliases.cachedir, f'fb_{region}.pkl')
#         self.h5path = os.path.join(repopath, 'cache', 'all.h5')
        self.timezone = self.TZS[region]

    @property
    def daskfrm(self):
        try:
            return self._daskfrm
        except AttributeError:
            frm = daskdf.read_csv(f"{self.datadir}/*.csv")
            frm = frm.rename(columns = self.RENAMEDICT)[self.KEEPKEYS]
            for key, func in self.PROCFUNCS.items():
                frm[key] = frm[key].astype(func)
            self._daskfrm = frm
            return frm

    @property
    def datafilenames(self):
        datafilenames = pd.Series(sorted(
            os.path.basename(fname)
                for fname in glob(os.path.join(self.datadir, "*.csv"))
                    if not fname == 'all.csv'
            ))
        datetimes = datafilenames.apply(datafilename_to_datetime)
        return pd.DataFrame(
            zip(datetimes, datafilenames),
            columns = ['datetimes', 'datafilenames']
            ).set_index('datetimes')['datafilenames']

    def make_blank(self):
        return pd.DataFrame(
            columns = (keys := self.KEEPKEYS)
            ).set_index(keys[:-1])['n']
    @property
    def frm(self):
        try:
            return self._prefrm
        except AttributeError:
            try:
                out = pd.read_pickle(self.prepath)
            except FileNotFoundError:
                out = self.make_blank()
            self._prefrm = out
            return out

    def load_fbcsv(self, dtime):
        seed = int.from_bytes(str(dtime).encode(), byteorder = 'big')
        rng = np.random.default_rng(seed = seed)
        fullpath = os.path.join(self.datadir, self.datafilenames.loc[dtime])
        frm = pd.read_csv(fullpath)
        frm = frm.rename(mapper = self.RENAMEDICT, axis = 1)
        frm = frm[self.KEEPKEYS]
        frm['n'].fillna(0, inplace = True)
        for key, func in self.PROCFUNCS.items():
            frm[key] = frm[key].apply(func)
        frm['n'].where(
            frm['n'] > 0,
            np.round(np.sqrt(rng.random(len(frm))) * 8 + 1),
            inplace = True
            )
        frm['datetime'] = dtime
        frm['datetime'] = frm['datetime'].dt.tz_convert(self.timezone)
        frm['km'] = frm['km'] > 0
        frm = frm.set_index(['datetime', 'quadkey', 'end_key', 'km'])
        frm = frm.sort_index()
        frm = frm['n']
        frm = frm.groupby(level = frm.index.names).sum()
        return frm

    def update_frm(self, new):
        frm = self.frm
        frm = pd.concat([frm, new]).sort_index()
        self._prefrm = frm
        frm.to_pickle(self.prepath)

    @property
    def datesloaded(self):
        return self.frm.index.levels[0]
    @property
    def datesdisk(self):
        return self.datafilenames.index
    @property
    def datesnotloaded(self):
        return [key for key in self.datesdisk if not key in self.datesloaded]

    def process_arg(self, arg):
        if isinstance(arg, datetime):
            return arg
        if isinstance(arg, str):
            return process_datetime(arg)
        if isinstance(arg, int):
            return self.datafilenames.index[arg]
        if arg is None:
            return None
        if isinstance(arg, Sequence):
            return arg
        if isinstance(arg, slice):
            start, stop, step = (
                self.process_arg(val)
                    for val in (arg.start, arg.stop, arg.step)
                )
            return slice(start, stop, step)
        if arg is Ellipsis:
            return arg
        raise ValueError(type(arg))
        
#             return self.process_arg(datetime_str_from_datafilename(arg))

    def __getitem__(self, arg):
        arg = self.process_arg(arg)
        if isinstance(arg, datetime):
            frm = self.frm
            try:
                return frm.loc[arg]
            except KeyError:
                new = self.load_fbcsv(arg)
                self.update_frm(new)
                return self.frm.loc[arg]
        if isinstance(arg, Sequence):
            print(f"Loading many files from '{self.region}'")
            news = []
            maxi = len(arg)
            datesnotloaded = self.datesnotloaded
            for i, subarg in enumerate(arg):
                if subarg in datesnotloaded:
                    new = self.load_fbcsv(subarg)
                    news.append(new)
                update_progressbar(i, maxi)
            if news:
                new = pd.concat(news)
                self.update_frm(new)
            print("\nDone.")
            return self.frm.loc[arg]
        if isinstance(arg, slice):
            return self[sorted(self.datafilenames.loc[arg].index)]
        if arg is Ellipsis:
            datesnotloaded = self.datesnotloaded
            if datesnotloaded:
                _ = self[self.datesnotloaded]
            return self.frm
        assert False

@lru_cache
def get_fb_loader(region):
    return FBDataset(region)
    
def load_fb(region, sample = ...):
    return get_fb_loader(region)[sample]


def download_zip(zipurl, destination):
    r = requests.get(zipurl)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(destination)

@lru_cache
def load_googlenames():
    path = os.path.join(aliases.resourcesdir, 'googlenames.pkl')
    with open(path, mode = 'rb') as file:
        return pickle.load(file)

@lru_cache
def load_google(region = None, update = False):

    if update:
        download_zip(
            ("https://www.gstatic.com/covid19/mobility/"
            "Region_Mobility_Report_CSVs.zip"),
            os.path.join(aliases.datadir, 'google')
            )

    frm2020 = pd.read_csv(os.path.join(
        aliases.datadir, 'google', '2020_AU_Region_Mobility_Report.csv'
        ))
    frm2021 = pd.read_csv(os.path.join(
        aliases.datadir, 'google', '2021_AU_Region_Mobility_Report.csv'
        ))

    frm = pd.concat([frm2020, frm2021])
    frm = frm.drop(
        ['country_region_code', 'country_region', 'place_id',
         'census_fips_code', 'metro_area', 'iso_3166_2_code'],
        axis = 1
        )
    frm = frm.dropna()
    frm = frm.rename(dict(
        sub_region_1 = 'state', sub_region_2 = 'name'
        ), axis = 1)
    frm['date'] = frm['date'].apply(pd.to_datetime)

    googlenames = load_googlenames()
    frm['name'] = frm['name'].apply(googlenames.__getitem__)

    if region is None:
        frm = frm.set_index(['date', 'state', 'name'])
    else:
        global STATENAMES, GCCNAMES, GCCSTATES
        ismetro = region in GCCNAMES
        state = STATENAMES[region if not ismetro else GCCSTATES[region]]
        frm = frm.loc[frm['state'] == state].drop('state', axis = 1)
        frm = frm.set_index(['date', 'name'])
        if ismetro:
            sa = load_sa(4, region)
            gcc = sa.unary_union
            lgas = load_lgas()
            inters = lgas.within(gcc)
            inters = sorted(inters.loc[inters].index)
            frm = frm.loc[(slice(None), inters),]
    frm = frm.sort_index()

    dupeinds = frm.index.duplicated(False)
    duplicateds = frm.iloc[dupeinds].sort_index()
    agg = duplicateds.groupby(level = frm.index.names).mean()
    frm = pd.concat([frm.loc[~dupeinds], agg])
    frm = frm.sort_index()

    frm = frm.rename(dict(zip(
        frm.columns,
        (
            strn.removesuffix('_percent_change_from_baseline')
                for strn in frm.columns
            ),
        )), axis = 1)

    return frm


class ABSSA:

    STATENAMES = STATENAMES

    GCCNAMES = GCCNAMES

    __slots__ = ('_frm', 'name', 'filename', 'level', 'region', '_region')

    def __init__(self, level = 2, region = None):
        self.level = level
        if not 2 <= level <= 4:
            raise ValueError
        if region is None:
            self.region = None
            self._region = None
        else:
            self.region = region
            if region in (states := self.STATENAMES):
                self._region = ('state', states[region])
            elif region in (gccs := self.GCCNAMES):
                self._region = ('gcc', gccs[region])
        name = self.name = f"abs_sa_{level}"
        if not region is None:
            name += '_' + region
        self.filename = os.path.join(aliases.cachedir, name) + '.pkl'

    def get_frm(self):
        try:
            return self._frm
        except AttributeError:
            pass
        try:
            return pd.read_pickle(self.filename)
        except FileNotFoundError:
            pass
        out = self.make_frm()
        self._frm = out
        out.to_pickle(self.filename)
        return out

    def make_basic_frm(self):

        frm = gpd.read_file(os.path.join(
            aliases.resourcesdir,
            'SA2_2016_AUST.shp'
            ))

        pop = pd.read_csv(os.path.join(
            aliases.resourcesdir,
            'ABS_ANNUAL_ERP_ASGS2016_29062021113341414.csv',
            ))
        pop = pop.loc[pop['REGIONTYPE'] == 'SA2']
        pop = pop.loc[pop['Region'] != 'Australia']
        pop = pop.loc[pop['Region'] != 'Australia']
        pop = pop[['Region', 'Value']].set_index('Region')['Value']

        frm['pop'] = frm['SA2_NAME16'].apply(lambda x: pop.loc[x])

        frm = frm.drop('SA2_5DIG16', axis = 1)
        frm = frm.rename(dict(
            AREASQKM16 = 'area',
            SA2_MAIN16 = 'SA2_CODE16',
            GCC_NAME16 = 'gcc',
            STE_NAME16 = 'state',
            ), axis = 1)

        frm = frm.dropna()

        if not (region := self._region) is None:
            colname, val = region
            frm = frm.loc[frm[colname] == val]

        return frm

    def make_frm(self):

        frm = self.make_basic_frm()

        level = self.level
        for i in range(3, min(level + 1, 5)):
            dropcols = [
                col for col in frm.columns if col.startswith(f"SA{i - 1}")
                ]
            frm = frm.drop(dropcols, axis = 1)
            aggfuncs = dict(
                geometry = shapely.ops.unary_union,
                pop = sum,
                area = sum,
                )
#             passkeys = [
#                 *[key for key in frm.columns if key.startswith('GCC')],
#                 *[key for key in frm.columns if key.startswith('STE')],
#                 ]
#             aggfuncs.update({key: lambda x: x.iloc[0] for key in passkeys})
            agg = frm.groupby(f'SA{i}_CODE16').aggregate(aggfuncs)
            frm = frm.set_index(f'SA{i}_CODE16')
            frm[list(aggfuncs)] = agg
            frm = frm.drop_duplicates()
            frm = frm.reset_index()
        keepcols = ['geometry', 'pop', 'area', 'gcc', 'state']
        keepcols.extend(
            col for col in frm.columns if col.startswith(f'SA{level}')
            )
        frm = frm[keepcols]
        frm = frm.rename({
            f"SA{level}_CODE16": 'code',
            f"SA{level}_NAME16": 'name',
            }, axis = 1)
        frm['code'] = frm['code'].astype(int)
        frm = frm.set_index('name')

        if not (region := self.region) is None:
            frm = frm.drop('state', axis = 1)
            if region in self.GCCNAMES:
                frm = frm.drop('gcc', axis = 1)
        
        return frm

    @property
    def frm(self):
        return self.get_frm()

@lru_cache
def get_sa_loader(level, region = None):
    return ABSSA(level, region)
    
def load_sa(level, region = None):
    return get_sa_loader(level, region).frm

# def load_generic(code):
#     if code.startswith('sa'):
#         return load_sa(int(code[-1]))


def load_lgas():
    paths = [aliases.resourcesdir, 'LGA_2019_AUST.shp']
    lgas = gpd.read_file(os.path.join(*paths))
    lgas = lgas.dropna()
    lgas = lgas.rename(dict(
        LGA_NAME19 = 'name',
        STE_NAME16 = 'state',
        AREASQKM19 = 'area',
        LGA_CODE19 = 'code',
        ), axis = 1)
    lgas['code'] = lgas['code'].astype(int)
    lgas = lgas[['name', 'code', 'state', 'area', 'geometry']]
    lgas = lgas.set_index('name')
    pops = pd.read_csv(os.path.join(
        aliases.resourcesdir,
        "ABS_ERP_LGA2020_15072021114736834.csv"
        ))
    pops = pops[['Region', 'Value']].set_index('Region')['Value']
    lgas['pop'] = pops
    lgas = lgas.reset_index().set_index('name')
    return lgas


def load_seifa():

    seifa = pd.read_csv(
        os.path.join(aliases.resourcesdir, '2033055001 - lga indexes - Table 1.csv')
        )

    seifa = seifa.dropna()
    seifa.columns = [
        'code',
        'name',
        'Index of Relative Socio-economic Disadvantage - Score',
        'Index of Relative Socio-economic Disadvantage - Decile',
        'Index of Relative Socio-economic Advantage and Disadvantage - Score',
        'Index of Relative Socio-economic Advantage and Disadvantage - Decile',
        'Index of Economic Resources - Score',
        'Index of Economic Resources - Decile',
        'Index of Education and Occupation - Score',
        'Index of Education and Occupation - Decile',
        'pop',
        ]

    for column in {
            'Index of Relative Socio-economic Disadvantage - Score',
            'Index of Relative Socio-economic Disadvantage - Decile',
            'Index of Relative Socio-economic Advantage and Disadvantage - Score',
            'Index of Relative Socio-economic Advantage and Disadvantage - Decile',
            'Index of Economic Resources - Score',
            'Index of Economic Resources - Decile',
            'Index of Education and Occupation - Score',
            'Index of Education and Occupation - Decile',
            }:
        seifa[column] = seifa[column].apply(
            lambda x: int(x) if x.isnumeric() else None
            )
        seifa = seifa.dropna()
        seifa[column] = seifa[column].astype(int)

    seifa = seifa.set_index('name')

    return seifa


###############################################################################
''''''
###############################################################################
