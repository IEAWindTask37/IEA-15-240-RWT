"""Helper functions for making the tower/monopile models.

TODO: Move this to a top-level package so it can be reused in all subfolders and
also tests.
"""
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import yaml


def load_yaml(path):
    """Load yaml file to dictionary."""
    with open(path, 'r', encoding='utf-8') as stream:
        try:
            res = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return res


def load_body_properties(bodyname, yamldict, start_from_zero=False):
    """Get body geometry and material properties from yaml data.

    tower station, outer diameter, thickness, E, G, rho.
    """
    # body dictionary
    bodydict = yamldict['components'][bodyname]
    # get tower dimensions
    twr_stn = np.array(bodydict['outer_shape_bem']['reference_axis']['z']['values'])
    if start_from_zero:
        twr_stn -= twr_stn[0]
    outfitting_factor = bodydict['internal_structure_2d_fem']['outfitting_factor']
    out_diam = np.array(bodydict['outer_shape_bem']['outer_diameter']['values'])
    wall = bodydict['internal_structure_2d_fem']['layers'][0]
    assert wall['name'] == bodyname + '_wall'
    thick = np.array(wall['thickness']['values'])
    # get material properties
    material = wall['material']
    mat_props = [d for d in yamldict['materials'] if d['name'] == material][0]
    E, G, rho = mat_props['E'], mat_props['G'], mat_props['rho']
    return twr_stn, out_diam, thick, E, G, rho, outfitting_factor


def load_elastodyn_distprop(path):
    """Load distributed properties in ED file."""
    with open(path, 'r', encoding='utf-8') as f:
        for il, line in enumerate(f):
            if il == 3:
                ntwinpst = int(line.split()[0])
                break
    ed_distprop = np.loadtxt(path, skiprows=19, max_rows=ntwinpst)
    return ed_distprop


def plot_od_thickness(stn, out_diam, thick, bodyname):
    """Plot outer diameter and thickness."""
    fig, (ax0, ax1) = plt.subplots(1, 2, num=1, figsize=(7, 3))

    ax0.plot(out_diam, stn)
    ax0.set(xlabel='Outer diameter [m]', ylabel='Height [m]')
    ax0.grid('on')

    ax1.plot(thick*1000, stn)
    ax1.set(xlabel='Wall thickness [mm]', ylabel='Height [m]')
    ax1.grid('on')

    fig.suptitle(f'Design for body "{bodyname}"')
    fig.tight_layout()

    return fig, (ax0, ax1)


def make_hawc2_st_array(stn, out_diam, thick, E, G, rho, outfitting_factor):
    """Make hawc2 st-file array."""
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

    return out_arr


def plot_dist_matprops_of_h2(out_arr, ed_st, bodyname):
    """Plot distributed material properties, OpenFAST vs HAWC2"""
    h2_stn = (out_arr[:, 0] - out_arr[0, 0])/(out_arr[-1, 0] - out_arr[0, 0])

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

    return fig, (ax0, ax1, ax2)


def save_h2_st_arr(path, out_arr, bodyname, notorsion=True, rigid=True):
    """Save hawc2 st file from array."""
    # flexible
    header1 = f'#1 {bodyname.capitalize()} made by automatic script on {date.today().strftime("%d-%b-%Y")}'
    header2 = '\t'.join(['r', 'm', 'x_cg', 'y_cg', 'ri_x', 'ri_y', 'x_sh', 'y_sh', 'E', 'G',
                    'I_x', 'I_y', 'I_p', 'k_x', 'k_y', 'A', 'pitch', 'x_e', 'y_e'])
    header = '\n'.join([header1, header2, f'$1 {out_arr.shape[0]} Flexible'])
    fmt = ['%.3f'] + ['%.4E'] * 18
    np.savetxt(path, out_arr, delimiter='\t', fmt=fmt, header=header, comments='')
    # flexible (no torsion)
    if notorsion:
        out_arr[:, 9] *= 1e7  # increase G
        header = '\n'.join([header2, f'$2 {out_arr.shape[0]} Flexible (no torsion)'])
        with open(path, 'a', encoding='utf-8') as f:
            np.savetxt(f, out_arr, delimiter='\t', fmt=fmt, header=header, comments='')
    # rigid
    if rigid:
        out_arr[:, 8] *= 1e7  # increase E
        header = '\n'.join([header2, f'$3 {out_arr.shape[0]} Rigid'])
        with open(path, 'a', encoding='utf-8') as f:
            np.savetxt(f, out_arr, delimiter='\t', fmt=fmt, header=header, comments='')

    print(f'St-file saved to "{path}"')
