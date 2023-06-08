"""Make the HAWC2 tower and monopile files from the yaml

Note that HAWC2 needs two separate body files for the monopile: one for the embedded portion
and one for the non-embedded portion.

Requirements: numpy, matplotlib, pyyaml
"""
from pathlib import Path

import matplotlib.pyplot as plt

import _functions as myf


save_st = 1  # save the HAWC2 tower and monopile st files?

# paths
h2dir = Path(__file__).parents[1]  # directory of hawc2 model (one level above)
yaml_path = h2dir / '../../WT_Ontology/IEA-15-240-RWT.yaml'  # yaml file with turbine definition
ed_paths = dict(tower=h2dir / '../../OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_ElastoDyn_tower.dat',
                monopile=h2dir / '../../OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_ElastoDyn_tower.dat',
                embeddedmonopile=h2dir / '../../OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_ElastoDyn_tower.dat',
                )  # elastodyn file

# constants
MUDLINE = -30
TWR_START = 15

# load the yaml file as nested dictionaries
yamldict = myf.load_yaml(yaml_path)

for bodyname in ['tower', 'embeddedmonopile', 'monopile']:

    # get dimensions and material properties
    yamlname = 'monopile' if 'monopile' in bodyname else 'tower'
    stn, out_diam, thick, E, G, rho, outfitting_factor = myf.load_body_properties(yamlname, yamldict, start_from_zero=False)

    # isolate/tweak data if we are monopile or embedded monopile
    if bodyname == 'monopile':
        mask = (stn > MUDLINE) & (stn <= TWR_START)  # from -30 (excl) to +15 (incl)
        stn = stn[mask]
        out_diam = out_diam[mask]
        thick = thick[mask]
        stn[0] = MUDLINE  # correct first station from -29.999 to -30
    elif bodyname == 'embeddedmonopile':
        mask = (stn <= MUDLINE)   # from -75 (incl) to -30 (incl)
        stn = stn[mask]
        out_diam = out_diam[mask]
        thick = thick[mask]

    # plot outer diameter and thickness for verification
    myf.plot_od_thickness(stn, out_diam, thick, bodyname)

    # make hawc2 array
    out_arr = myf.make_hawc2_st_array(stn, out_diam, thick, E, G, rho, outfitting_factor)

    # compare results with elastodyn
    ed_st = myf.load_elastodyn_distprop(ed_paths[bodyname])

    # plot distributed material properties, ed versus hawc2
    myf.plot_dist_matprops_of_h2(out_arr, ed_st, bodyname)

    if save_st:
        h2_st_path = h2dir / f'data/IEA_15MW_RWT_{bodyname.capitalize()}_st.dat'
        myf.save_h2_st_arr(h2_st_path, out_arr, bodyname)

    plt.show()
