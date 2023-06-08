"""Make the HAWC2 tower and monopile files from the yaml

Note that HAWC2 needs two separate body files for the monopile: one for the embedded portion
and one for the non-embedded portion.

Requirements: numpy, matplotlib, pyyaml
"""
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

import _functions as myf


save_st = 0  # save the HAWC2 tower and monopile st files?

# paths
h2dir = Path(__file__).parents[1]  # directory of hawc2 model (one level above)
yaml_path = h2dir / '../../WT_Ontology/IEA-15-240-RWT.yaml'  # yaml file with turbine definition
ed_paths = dict(tower=h2dir / '../../OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_ElastoDyn_tower.dat',
                monopile=h2dir / '../../OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_ElastoDyn_tower.dat')  # elastodyn file

# load the yaml file as nested dictionaries
yamldict = myf.load_yaml(yaml_path)

for bodyname in ['tower', 'embeddedmonopile', 'monopile']:

    # get dimensions and material properties
    stn, out_diam, thick, E, G, rho, outfitting_factor = myf.load_body_properties(bodyname, yamldict, start_from_zero=False)

    # plot outer diameter and thickness for verification
    myf.plot_od_thickness(stn, out_diam, thick, bodyname)

    # make hawc2 array
    out_arr = myf.make_hawc2_st_array(stn, out_diam, thick, E, G, rho, outfitting_factor)

    # compare results with elastodyn
    ed_st = myf.load_elastodyn_distprop(ed_paths[bodyname])

    # plot distributed material properties, ed versus hawc2
    myf.plot_dist_matprops_of_h2(out_arr, ed_st, bodyname)

    plt.show()

    if save_st:
        h2_st_path = h2dir / f'data/IEA_15MW_RWT_{bodyname.capitalize()}_st.dat'
        myf.save_h2_st_arr(h2_st_path, out_arr, bodyname)

    plt.show()
