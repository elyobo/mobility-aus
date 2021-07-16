###############################################################################
###############################################################################


from functools import partial

import pandas as pd
idx = pd.IndexSlice

import get
import load

import aliases
import analysis

from everest import window


def melsummary_plot():

    frm = get.get_aggregated_data('vic', aggtype = 'sa4', refresh = True)
    metro = load.load_sa(4, 'mel')
    frm = frm.loc[idx[:], metro.index]
    # frm = frm.loc['2021-01-01':]

    mobprops = analysis.mobile_proportion(frm).dropna()
    scores = mobprops.groupby(level = 'start') \
        .apply(partial(analysis.calculate_day_scores, n = 8)) \
        .dropna()

    covid = analysis.get_gov_covid_data()
    covid = covid['new'].groupby(level = 'date').sum()
    covid = covid.rolling(7).mean().dropna()

    canvas = window.Canvas(size = (18, 4.5))
    ax = canvas.make_ax()
    data = scores.loc[206]
    dates = data.index.values
    ax.line(
        window.DataChannel(
            data.index.values,
            label = "!$Date",
            ),
        window.DataChannel(
            data.values, lims = (-1, 2), capped = (True, True),
            label = "!$Mobility score"
            ),
        )
    for date, label, points in analysis.MELVIC_ANNOTATIONS:
        x = pd.Timestamp(date)
        y = data.loc[date]
        ax.annotate(
            x, y, "!$" + label,
            points = points, arrowProps = dict(arrowstyle = '->'),
            )

    ax2 = canvas.make_ax()
    ax2.line(
        window.DataChannel(
            covid.index.values, lims = (min(dates), max(dates)),
            ),
        window.DataChannel(
            covid.values / metro['pop'].sum() * 10000,
            lims = (0, 1),
            label = "!$New cases per 10,000 people (7-day average)"
            ),
        color = 'red',
        )
    ax2.props.edges.y.swap()
    ax2.props.grid.toggle()

    return canvas


###############################################################################
###############################################################################
