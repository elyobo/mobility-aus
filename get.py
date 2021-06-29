###############################################################################
###############################################################################

import os

import pandas as pd

import aliases
import aggregate
import load


def get_aggregated_data(region, aggtype = 'lga', sample = ..., refresh = False):
    print(f"Getting data for {region} aggregated to {aggtype}...")
    samstr = '_' + str(sample) if not sample is Ellipsis else ''
    name = f"aggregated_{region}_{aggtype}{samstr}.pkl"
    outpath = os.path.join(aliases.cachedir, name)
    if refresh:
        print("Generating...")
    else:
        print("Loading...")
        try:
            out = pd.read_pickle(outpath)
            print("Loaded.")
            print("Done.")
            return out
        except FileNotFoundError:
            print("Load failed. Generating...")   
    out = aggregate.aggregate(
        load.FBDataset(region)[sample],
        aggtype,
        region,
        )
    out.to_pickle(outpath)
    print("Done.")
    return out


###############################################################################
###############################################################################
