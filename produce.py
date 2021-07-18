###############################################################################
###############################################################################


from functools import lru_cache

import load
import analysis


@lru_cache
def google_score(region, n = 4):
    frm = load.load_google(region)
    frm = analysis.make_scorefrm(frm, region, n = n)
    return 1. - frm['residential']


###############################################################################
###############################################################################
