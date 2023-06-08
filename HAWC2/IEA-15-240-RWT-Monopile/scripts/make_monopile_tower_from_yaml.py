"""Make the HAWC2 tower files from the yaml

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

for bodyname in ['tower', 'monopile']:

    # get dimensions and material properties
    stn, out_diam, thick, E, G, rho, outfitting_factor = myf.load_body_properties(bodyname, yamldict, start_from_zero=False)

    # plot outer diameter and thickness for verification
    fig, (ax0, ax1) = plt.subplots(1, 2, num=1, figsize=(7, 3))

    ax0.plot(out_diam, stn)
    ax0.set(xlabel='Outer diameter [m]', ylabel='Height [m]')
    ax0.grid('on')

    ax1.plot(thick*1000, stn)
    ax1.set(xlabel='Wall thickness [mm]', ylabel='Height [m]')
    ax1.grid('on')

    fig.suptitle(f'Design for body "{bodyname}"')
    fig.tight_layout()

    # intermediates
    r_out = out_diam / 2  # outer diameter [m]
    r_in = r_out - thick  # inner diameter [m]

    area = pi * (r_out**2 - r_in**2)  # cross-sectional area [m^2]
    mom_iner = pi/4 * (r_out**4 - r_in**4)  # area moment of inertia [m^4]
    tors_const = pi/2 * (r_out**4 - r_in**4)  # torsional constant [m^4]
    shear_red = 0.5 + 0.75 * thick / r_out  # shear reduction factor [-]

    # create the array of values to write
    nstn = 19
    out_arr = np.full((stn.size, nstn), np.nan)
    out_arr[:, 0] = stn  # station
    out_arr[:, 1] = rho*area*outfitting_factor  # mass/length
    out_arr[:, 2] = 0  # center of mass, x
    out_arr[:, 3] = 0  # center of mass, y
    out_arr[:, 4] = np.sqrt(mom_iner/area)  # radius gyration, x
    out_arr[:, 5] = np.sqrt(mom_iner/area)  # radius gyration, y
    out_arr[:, 6] = 0  # shear center, x
    out_arr[:, 7] = 0  # shear center, y
    out_arr[:, 8] = E  # young's modulus
    out_arr[:, 9] = G  # shear modulus
    out_arr[:, 10] = mom_iner  # area moment of inertia, x
    out_arr[:, 11] = mom_iner  # area moment of inertia, y
    out_arr[:, 12] = tors_const  # torsional stiffness constant
    out_arr[:, 13] = shear_red  # shear reduction, x
    out_arr[:, 14] = shear_red  # shear reduction, y
    out_arr[:, 15] = area  # cross-sectional area
    out_arr[:, 16] = 0  # structural pitch
    out_arr[:, 17] = 0  # elastic center, x
    out_arr[:, 18] = 0  # elastic center, y

    # compare results with elastodyn
    ed_st = myf.load_elastodyn_distprop(ed_paths[bodyname])
    h2_stn = (out_arr[:, 0] - out_arr[0, 0])/(out_arr[-1, 0] - out_arr[0, 0])

    # visualize the difference
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, num=2, figsize=(7, 3))

    ax0.plot(ed_st[:, 1], ed_st[:, 0], label='OpenFAST')
    ax0.plot(out_arr[:, 1], h2_stn, '--', label='HAWC2')  # mpl
    ax0.set(label='Mass density [m]', ylabel='Height [m]')
    ax0.grid('on')
    ax0.legend()

    ax1.plot(ed_st[:, 2], ed_st[:, 0])
    ax1.plot(out_arr[:, 8]*out_arr[:, 10], h2_stn, '--', label='HAWC2')  # EIxx
    ax1.set(xlabel='TwFAStif [mm]')
    ax1.grid('on')

    ax2.plot(ed_st[:, 3], ed_st[:, 0])
    ax2.plot(out_arr[:, 8]*out_arr[:, 11], h2_stn, '--', label='HAWC2')  # EIyy
    ax2.set(xlabel='TwSSStif [mm]')
    ax2.grid('on')

    fig.suptitle(f'Distributed material properties for body "{bodyname}"')
    fig.tight_layout()

    plt.show()

    if save_st:
        h2_st_path = h2dir / f'data/IEA_15MW_RWT_{bodyname.capitalize()}_st.dat'
        # flexible
        header1 = f'#1 {bodyname.capitalize()} made by automatic script on {date.today().strftime("%d-%b-%Y")}'
        header2 = '\t'.join(['r', 'm', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'x_sh', 'y_sh', 'E', 'G',
                        'I_x', 'I_y', 'I_p', 'k_x', 'k_y', 'A', 'pitch', 'x_e', 'y_e'])
        header = '\n'.join([header1, header2, f'$1 {out_arr.shape[0]} Flexible'])
        fmt = ['%.3f'] + ['%.4E'] * 18
        np.savetxt(h2_st_path, out_arr, delimiter='\t', fmt=fmt, header=header, comments='')
        # flexible (no torsion)
        out_arr[:, 9] *= 1e7  # increase G
        header = '\n'.join([header2, f'$2 {out_arr.shape[0]} Flexible (no torsion)'])
        with open(h2_st_path, 'a', encoding='utf-8') as f:
            np.savetxt(f, out_arr, delimiter='\t', fmt=fmt, header=header, comments='')
        # rigid
        out_arr[:, 8] *= 1e7  # increase E
        header = '\n'.join([header2, f'$3 {out_arr.shape[0]} Rigid'])
        with open(h2_st_path, 'a', encoding='utf-8') as f:
            np.savetxt(f, out_arr, delimiter='\t', fmt=fmt, header=header, comments='')
        print(f'St-file saved to "{h2_st_path}"')

    plt.show()
