###############################################################################
###############################################################################


import re

import pandas as pd

import aliases


def mobile_proportion(frm):
    nsums = frm.groupby(level = ('date', 'start')).sum()
    xs = frm.xs(True, level = 'km')
    mobsums = xs.groupby(level = ('date', 'start')).sum()
    return mobsums / nsums

def calculate_day_scores(series, level = 'date', n = 4):
    '''
    Takes a series indexed by date
    and returns normalised values grouped by date.
    '''
    index = series.index.get_level_values(level)
    series = pd.DataFrame(data = dict(
        val = series.values,
        date = index,
        day = [int(d.strftime('%w')) for d in index.tolist()]
        )).set_index([level, 'day'])['val']
    groups = series.groupby(level = 'day')
    highs = groups.apply(lambda s: s.nlargest(n).mean())
    lows = groups.apply(lambda s: s.nsmallest(n).mean())
    series = (series - lows) / (highs - lows)
    series = pd.Series(series.values, index)
    return series

MELVIC_ANNOTATIONS = [
#     ('2020-04-25', 'Anzac Day', (0, -30)),
#     ('2020-05-13', 'Easing', (0, -30)),
    ('2020-05-31', 'Cafes\nreopen', (0, 30)),
    ('2020-06-08', "Queen's\nBirthday", (0, -30)),
    ('2020-06-26', 'School\nholidays', (-30, 30)),
    ('2020-06-30', 'Postcode\nlockdowns', (-15, -45)),
    ('2020-07-08', 'Stage 3', (0, 30)),
    ('2020-07-19', 'Mask\nmandate', (0, -30)),
    ('2020-08-02', 'Stage 4', (0, 30)),
    ('2020-08-06', 'Business\nclosures', (0, -30)),
    ('2020-09-06', 'Roadmap\nplan', (-15, 60)),
    ('2020-09-13', 'First\nStep', (0, 30)),
    ('2020-09-27', 'Second\nStep', (0, -30)),
    ('2020-10-11', 'Picnics\nallowed', (-30, 30)),
    ('2020-10-18', 'Travel\nrelaxed', (0, 30)),
    ('2020-10-23', 'Grand Final\nholiday', (0, -30)),
    ('2020-10-28', 'Third\nStep', (0, 30)),
    ('2020-11-03', 'Cup Day', (0, -30)),
    ('2020-11-08', 'Ring of Steel\nends', (0, 45)),
    ('2020-11-22', 'Last\nStep', (0, -30)),
    ('2020-12-06', 'COVIDSafe\nSummer', (0, 45)),
    ('2020-12-25', 'Christmas\nDay', (-30, -30)),
    ('2020-12-26', 'Boxing\nDay', (0, 30)),
    ('2021-01-01', "New Year's\nDay", (0, -30)),
    ('2021-01-26', "National\nholiday", (0, -30)),
    ('2021-02-13', "Circuit\nbreaker", (0, -45)),
    ('2021-03-08', 'Labour\nDay', (0, -30)),
    ('2021-04-02', 'Easter', (0, -30)),
    ('2021-04-25', 'Anzac Day', (0, -45)),
    ('2021-05-11', 'Wollert\ncluster', (0, 30)),
    ('2021-05-28', 'Fourth\nlockdown', (-15, -30)),
    ('2021-06-11', 'Easing', (-15, 30)),
    ('2021-06-14', "Queen's\nBirthday", (15, -30)),
    ]

def remove_brackets(x):
    # Remove brackets from ABS council names:
    return re.sub("[\(\[].*?[\)\]]", "", x).strip()

def detect_american_dates(dates):
    months = sorted(set([date.split('-')[0] for date in dates]))
    return len(months) <= 12

def to_american_date(datestr):
    day, month, year = datestr.split('-')
    return '/'.join((month, day, year))

def get_gov_covid_data(agg = 'lga', region = 'vic'):
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
    cases['name'] = cases['name'].apply(remove_brackets)
    cases['mystery'] = cases['source'] == 'Acquired in Australia, unknown source'
    dropagg = tuple(v for v in aggchoices.values() if not v == agg)
    cases = cases.drop(['source', *dropagg], axis = 1)
    cases = cases.sort_values(['date', agg])
    cases['new'] = 1
    cases['mystery'] = cases['mystery'].apply(int)
    cases = cases.groupby(['date', agg])[['new', 'mystery']].sum()
    return cases

def make_casesFrm_gov(region = 'vic', agg = 'lga'):

    cases = get_gov_covid_data()

    names = list(set(cases.index.get_level_values('name')))
    base = datetime.datetime(2020, 1, 1)
    days = []
    day = base
    maxday = cases.index.get_level_values('date').max() + datetime.timedelta(days = 30)
    while day < maxday:
        days.append(day)
        day += datetime.timedelta(days = 1)
    blank = pd.DataFrame(
        itertools.product(days, names, [0], [0]),
        columns = ('date', 'name', 'new', 'mystery')
        )
    blank = blank.set_index(['date', 'name'])

    blank[cases.columns] = cases
    cases = blank
    cases = cases.fillna(0)

    lookup = make_sub_lookupFrm(region, 'lga')
    popDict = dict(zip(lookup.index, lookup['pop']))
    cases = cases.loc[(slice(None), popDict.keys()),]
    cases['pop'] = [popDict[n] for n in cases.index.get_level_values('name')]
    cases['new'] = cases['new'] / cases['pop'] * 10000
    cases['new_rolling'] = cases['new'].groupby(level = 'name', group_keys = False) \
        .rolling(7).mean().sort_index()
    cases['new_rolling'] = cases['new_rolling'].apply(lambda s: 0 if s < 1e-3 else s)
    cases['cumulative'] = cases.groupby('name')['new'].cumsum()
    cases = cases.dropna()
    cases = cases.drop('pop', axis = 1)

    return cases


###############################################################################
###############################################################################
