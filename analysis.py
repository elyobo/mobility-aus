###############################################################################
###############################################################################


import re
import datetime
import math
import itertools

from scipy import interpolate as sp_interp
import pandas as pd
import numpy as np


import aliases
import utils
import load


# def mobile_proportion(frm):
#     nsums = frm.groupby(level = ('date', 'start')).sum()
#     xs = frm.xs(True, level = 'km')
#     mobsums = xs.groupby(level = ('date', 'start')).sum()
#     return mobsums / nsums

PUBLICHOLIDAYS = dict(
    vic = dict((datetime.datetime(*date), label) for date, label in (
        ((2020, 1, 1), "New Year's Day"),
        ((2020, 1, 27), "Australia Day"),
        ((2020, 3, 9), "Labour Day"),
        ((2020, 4, 10), "Good Friday"),
        ((2020, 4, 11), "Easter Saturday"),
        ((2020, 4, 12), "Easter Sunday"),
        ((2020, 4, 13), "Easter Monday"),
        ((2020, 4, 25), "Anzac Day"),
        ((2020, 6, 8), "Queen's Birthday"),
        ((2020, 10, 23), "Friday before the AFL Grand Final"),
        ((2020, 11, 3), "Melbourne Cup"),
        ((2020, 12, 25), "Christmas Day"),
        ((2020, 12, 26), "Boxing Day"),
        ((2020, 12, 28), "Boxing Day Compensation"),
        ((2021, 1, 1), "New Year's Day"),
        ((2021, 1, 26), "Australia Day"),
        ((2021, 3, 8), "Labour Day"),
        ((2021, 4, 2), "Good Friday"),
        ((2021, 4, 3), "Easter Saturday"),
        ((2021, 4, 4), "Easter Sunday"),
        ((2021, 4, 5), "Easter Monday"),
        ((2021, 4, 25), "Anzac Day"),
        ((2021, 6, 14), "Queen's Birthday"),
        ((2021, 9, 24), "Friday before the AFL Grand Final"),
        ((2021, 11, 2), "Melbourne Cup"),
        ((2021, 12, 25), "Christmas Day"),
        ((2021, 12, 27), "Christmas Day Compensation"),
        ((2021, 12, 28), "Boxing Day Compensation"),
        )),
    nsw = dict((datetime.datetime(*date), label) for date, label in (
        ((2020, 1, 1), "New Year's Day"),
        ((2020, 1, 27), "Australia Day"),
        ((2020, 4, 10), "Good Friday"),
        ((2020, 4, 11), "Easter Saturday"),
        ((2020, 4, 12), "Easter Sunday"),
        ((2020, 4, 13), "Easter Monday"),
        ((2020, 4, 25), "Anzac Day"),
        ((2020, 6, 8), "Queen's Birthday"),
        ((2020, 10, 5), "Labour Day"),
        ((2020, 12, 25), "Christmas Day"),
        ((2020, 12, 26), "Boxing Day"),
        ((2020, 12, 28), "Boxing Day Compensation"),
        ((2021, 1, 1), "New Year's Day"),
        ((2021, 1, 26), "Australia Day"),
        ((2021, 4, 2), "Good Friday"),
        ((2021, 4, 3), "Easter Saturday"),
        ((2021, 4, 4), "Easter Sunday"),
        ((2021, 4, 5), "Easter Monday"),
        ((2021, 4, 25), "Anzac Day"),
        ((2021, 6, 14), "Queen's Birthday"),
        ((2021, 10, 4), "Labour Day"),
        ((2021, 12, 25), "Christmas Day"),
        ((2021, 12, 27), "Christmas Day Compensation"),
        ((2021, 12, 28), "Boxing Day Compensation"),
        )),
    )

COVIDMEASURES = dict(
    vic = dict((datetime.datetime(*date), label) for date, label in (
        ((2020, 3, 13), "First lockdown"),
        ((2020, 5, 13), "Easing"),
        ((2020, 5, 31), "Cafes reopen"),
        ((2020, 6, 30), "Postcode lockdowns"),
        ((2020, 7, 8), "Stage 3"),
        ((2020, 7, 18), "Mask mandate"),
        ((2020, 8, 2), "Stage 4"),
        ((2020, 8, 6), "Business closures"),
#         ((2020, 9, 6), "Roadmap plan"),
        ((2020, 9, 13), "First step"),
        ((2020, 9, 27), "Second step"),
        ((2020, 10, 11), "Picnics allowed"),
#         ((2020, 10, 18), "Travel relaxed"),
        ((2020, 10, 28), "Third step"),
        ((2020, 11, 8), "Ring of Steel ends"),
        ((2020, 11, 22), "Last step"),
        ((2020, 12, 6), "COVIDSafe Summer"),
        ((2021, 2, 15), "Circuit breaker"),
        ((2021, 5, 11), "Wollert cluster"),
        ((2021, 5, 28), "Fourth lockdown"),
        ((2021, 6, 11), "Easing"),
        ((2021, 7, 16), "Fifth lockdown"),
        ))
    )

ONEDAY = pd.Timedelta(1, unit = 'D')

def get_day(tstamp, hols):
    if tstamp in hols:
        return 7 # holiday
    day = tstamp.day_of_week
    if day == 0: # Monday
        global ONEDAY
        if tstamp + ONEDAY in hols:
            return 7
        return day
    if day == 4: # Friday
        if tstamp - ONEDAY in hols:
            return 7
        return day
    return day

def get_days(datetimes, region):
    global PUBLICHOLIDAYS
    state = (region if region in PUBLICHOLIDAYS else load.GCCSTATES[region])
    hols = PUBLICHOLIDAYS[state]
    return np.array([get_day(tstamp, hols) for tstamp in datetimes])

def calculate_averages(frm, level = 'date', weightKey = 'pop'):
    # Get a frame that contains averages by some chosen level
    serieses = dict()
    level = 'date'
    weightKey = 'pop'
    for key in [col for col in frm.columns if not col == weightKey]:
        fn = lambda f: np.average(f[key], weights = f[weightKey])
        series = frm[[key, weightKey]].groupby(level = level).apply(fn)
        serieses[key] = series
    return pd.DataFrame(serieses)

def calculate_day_scores(inp, region, n = 4):
    '''
    Takes a dataframe indexed by date
    and returns normalised values grouped by date.
    '''
    state = load.get_state(region)
    frm = inp.reset_index()
    frm['day'] = get_days(frm['date'], region)
    frm = frm.set_index([*inp.index.names, 'day'])
    procserieses = []
    for name, series in frm.iteritems():
        groups = series.groupby(
            level = [nm for nm in frm.index.names if not nm == 'date']
            )
        highs = groups.apply(lambda s: s.nlargest(n).median())
        lows = groups.apply(lambda s: s.nsmallest(n).median())
        series = ((series - lows) / (highs - lows)).clip(-1, 2)
        series = series.reset_index() \
            .set_index(inp.index.names).drop('day', axis = 1).sort_index()
        procserieses.append(series)
    frm = pd.concat(procserieses, axis = 1)
    if isinstance(inp, pd.Series):
        return frm[inp.name]
    return frm

def make_avfrm(frm):
    avfrm = calculate_averages(frm)
    avfrm['name'] = 'average'
    avfrm = avfrm.reset_index().set_index(frm.index.names)
    return avfrm

def make_seifaavs(frm):
    seifa = load.load_seifa()
    seifa = seifa['Index of Relative Socio-economic Disadvantage - Score']
    seifa = seifa.loc[set(frm.index.levels[1]).intersection(seifa.index)]
    lowSE = seifa.nsmallest(math.floor(len(seifa) / 3))
    highSE = seifa.nlargest(math.floor(len(seifa) / 3))
    midSE = seifa.loc[[
        key for key in seifa.index
            if not (key in lowSE.index or key in highSE.index)
        ]]
    seifaavs = []
    for name, se in zip(['lowSE', 'midSE', 'highSE'], [lowSE, midSE, highSE]):
        subnames = set(se.index).intersection(frm.index.levels[1])
        subfrm = frm.loc[(slice(None), subnames),]
        subavfrm = calculate_averages(subfrm)
        subavfrm['name'] = name
        subavfrm = subavfrm.reset_index().set_index(frm.index.names)
        seifaavs.append(subavfrm)
    return pd.concat(seifaavs)

def make_scorefrm(frm, region, n = 4):

    lgas = load.load_lgas()

    anfrm = calculate_day_scores(frm, region, n = n)
    anfrm['pop'] = list(lgas['pop'].loc[frm.index.get_level_values('name')])

    avfrm = make_avfrm(anfrm)
    seifafrm = make_seifaavs(anfrm)

    frm = pd.concat([anfrm, avfrm, seifafrm])
    frm = frm.drop('pop', axis = 1)
    frm = frm.dropna()
    frm = frm.sort_index()

    return frm


def detect_american_dates(dates):
    months = sorted(set([date.split('-')[0] for date in dates]))
    return len(months) <= 12

def to_american_date(datestr):
    day, month, year = datestr.split('-')
    return '/'.join((month, day, year))

def get_gov_covid_data(region = 'vic', agg = 'lga'):
    aggchoices = dict(lga = 'name', postcode = 'postcode')
    agg = aggchoices[agg]
    url = 'https://www.dhhs.vic.gov.au/ncov-covid-cases-by-lga-source-csv'
    cases = pd.read_csv(url)
#     cases['diagnosis_date'] = \
#         cases['diagnosis_date'].astype('datetime64[ns]')
    dates = cases['diagnosis_date']
    if not detect_american_dates(dates):
        dates = dates.apply(to_american_date)
    cases['diagnosis_date'] = dates.astype('datetime64[ns]')
    cases = cases.rename(dict(
        diagnosis_date = 'date',
        Localgovernmentarea = 'name',
        acquired = 'source',
        Postcode = 'postcode',
        ), axis = 1)
    cases = cases.loc[cases['source'] != 'Travel overseas']
    cases = cases.loc[cases['name'] != 'Overseas']
    cases = cases.loc[cases['name'] != 'Interstate']
    cases = cases.sort_index()
    cases['name'] = cases['name'].apply(utils.remove_brackets)
    cases['mystery'] = cases['source'] == \
        'Acquired in Australia, unknown source'
    dropagg = tuple(v for v in aggchoices.values() if not v == agg)
    cases = cases.drop(['source', *dropagg], axis = 1)
    cases = cases.sort_values(['date', agg])
    cases['new'] = 1
    cases['mystery'] = cases['mystery'].apply(int)
    cases = cases.groupby(['date', agg])[['new', 'mystery']].sum()
    return cases

def make_casesFrm_gov(region = 'vic', agg = 'lga'):

    if agg != 'lga':
        raise Exception("Not supported yet.")
    if region != 'vic':
        raise Exception("Not supported yet.")

    cases = get_gov_covid_data(region, agg)

    lgas = load.load_lgas()
    lganames = sorted(set(lgas.index))
    namesdict = dict(zip(
        (utils.remove_brackets(nm) for nm in lganames),
        lganames
        ))

    cases = cases.drop('Unknown', level = 'name')
    cases = cases.reset_index()
    cases['name'] = cases['name'].apply(namesdict.__getitem__)
    cases = cases.set_index(['date', 'name']).sort_index()

    base = datetime.datetime(2020, 1, 1)
    days = []
    day = base
    maxday = (
          cases.index.get_level_values('date').max()
        + datetime.timedelta(days = 30)
        )
    while day < maxday:
        days.append(day)
        day += datetime.timedelta(days = 1)
    names = sorted(set(cases.index.get_level_values('name')))
    blank = pd.DataFrame(
        itertools.product(days, names, [0], [0]),
        columns = ('date', 'name', 'new', 'mystery')
        )
    blank = blank.set_index(['date', 'name'])

    blank[cases.columns] = cases
    blank = blank.fillna(0)
    cases = blank

    popdict = dict(zip(
        lgas.index,
        lgas['pop']
        ))
    cases['pop'] = [popdict[nm] for nm in cases.index.get_level_values('name')]
    cases['new'] = cases['new'] / cases['pop'] * 10000
    cases['mystery'] = cases['mystery'] / cases['pop'] * 10000

    avfrm = make_avfrm(cases)
    seifafrm = make_seifaavs(cases)

    cases = pd.concat([cases, avfrm, seifafrm])

    cases = cases.drop('pop', axis = 1).sort_index()

    rolling = pd.concat([
        cases[['new', 'mystery']].xs(
                reg, level = 'name', drop_level = False
                ).rolling(7).mean()
            for reg in cases.index.levels[1]
        ]).dropna().sort_index()
    rolling = rolling.rename(
        dict(new = 'new_rolling', mystery = 'mystery_rolling'),
        axis = 1
        )
    cases[rolling.columns] = rolling

    cumulative = pd.concat([
        cases[['new', 'mystery']].xs(
                reg, level = 'name', drop_level = False
                ).cumsum()
            for reg in cases.index.levels[1]
        ]).dropna().sort_index()
    cumulative = cumulative.rename(
        dict(new = 'new_cumulative', mystery = 'mystery_cumulative'),
        axis = 1
        )
    cases[cumulative.columns] = cumulative

    cases = cases.dropna().sort_index()

    return cases


###############################################################################
###############################################################################
