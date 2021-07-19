###############################################################################
###############################################################################


from functools import lru_cache
import os

import aliases
import load
import analysis
from everest import window


@lru_cache
def google_score(region, n = 4):
    frm = load.load_google(region)
    frm = analysis.make_scorefrm(frm, region, n = n)
    frm = frm.rename(dict(residential = 'mobility'), axis = 1)
    return 1. - frm['mobility']

def two_cities():

    frms = dict()

    for region in ('syd', 'mel'):

        frm = google_score(region, n = 12)
        frms[region] = frm
        frm.to_csv(os.path.join(
            aliases.productsdir,
            f'{region}_google_analysis.csv'
            ))
        

    canvas = window.Canvas(size = (18, 4.5))
    ax = canvas.make_ax()

    for frm in frms.values():
        alldays = frm.xs('average', level = 'name')
        days = analysis.get_days(alldays.index, region)
        workdays = alldays.iloc[days < 5]
        ax.scatter(
            workdays.index,
            window.DataChannel(workdays.values, lims = (-1, 2)),
            s = 10.
            )
        ax.line(
            workdays.index,
            window.DataChannel(workdays.values, lims = (-1, 2)),
            )

    ax.props.legend.set_handles_labels(
        [row[0] for row in ax.collections[1::2]],
        ('Sydney', 'Melbourne')
        )
    ax.props.edges.y.label.text = '!$Mobility Score'
    ax.props.edges.x.label.text = '!$Date'
    ax.props.title.text = "!$A Tale of Two Cities: Diverging Lockdown Journeys"

    canvas.save('two_cities', aliases.productsdir)

    return canvas, frms

def melbourne_simple():

    region = 'mel'

    cases = analysis.make_casesFrm_gov()

    frm = google_score(region, n = 12)

    frm = frm.to_frame()
    frm['new'] = cases['new_rolling']

    canvas = window.Canvas(size = (18, 4.5))

    ax1 = canvas.make_ax()
    alldays = frm['mobility'].xs('average', level = 'name')
    days = analysis.get_days(alldays.index, region)
    workdays = alldays.iloc[days < 5]
    ax1.scatter(
        workdays.index,
        window.DataChannel(workdays.values, lims = (-1, 2)),
        s = 10.
        )
    ax1.line(
        workdays.index,
        window.DataChannel(workdays.values, lims = (-1, 2)),
        )
    ax1.props.edges.x.label.text = '!$Date'
    ax1.props.edges.y.label.text = '!$Mobility score'

    ax2 = canvas.make_ax()
    data = frm['new'].xs('average', level = 'name')
    ax2.line(
        data.index,
        data.values,
        color = 'red'
        )
    ax2.props.edges.x.visible = False
    ax2.props.edges.y.swap()
    ax2.props.edges.y.label.text = '!$New cases\n(7-day rolling average per 10,000 people)'

    data = frm['new'].xs('average', level = 'name')
    ax2.annotate(
        data.index[70],
        data.values[70],
        '!$ COVID cases',
        arrowProps = dict(arrowstyle = 'fancy', color = 'red'),
        points = (0, 50),
        c = 'red',
        )

    data = frm['mobility'].xs('average', level = 'name')
    ax1.annotate(
        data.index[70],
        data.values[70],
        '!$Mobility score',
        arrowProps = dict(arrowstyle = 'fancy', color = 'blue'),
        points = (0, 50),
        c = 'blue',
        )
    for i, (date, label) in enumerate(analysis.COVIDMEASURES['vic'].items()):
        if date not in data.index:
            continue
        label = label.replace(' ', '\n')
        ax1.annotate(
            date,
            data.loc[date],
            f"!${label}",
            arrowProps = dict(arrowstyle = '->'),
    #         points = (0, 30),
            points = (0, (-35 if i % 2 else 35)),
    #         points = (0, (30 if i % 2 else 60)),
    #         rotation = 30
            )

    ax1.props.title.text = "!$Lessons of COVID:\nHow Melbourne defeated its second wave - and prevented a third"

    canvas.save('melbourne_simple', aliases.productsdir)

    frm.to_csv(os.path.join(aliases.productsdir, 'melbourne_simple.csv'))

    return canvas, frm
    

def produce_all():
    two_cities()
    melbourne_simple()


###############################################################################
###############################################################################
