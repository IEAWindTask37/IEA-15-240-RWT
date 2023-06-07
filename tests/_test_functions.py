"""Helper functions for scripts for onshore tower analysis.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


FROOT = Path(__file__).parents[1]  # location of directory one level up (main GitHub)
PI = np.pi  # for convenience


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


def calculate_area(out_diam, thick):
    """Calculate area of tower."""
    r_out = out_diam / 2  # outer diameter [m]
    r_in = r_out - thick  # inner diameter [m]
    return PI * (r_out**2 - r_in**2)  # cross-sectional area [m^2]


def calculate_mpl(out_diam, thick, rho, outfitting_factor=1):
    """Calculate mass per unit length, including outfitting."""
    area = calculate_area(out_diam, thick)
    return rho * area * outfitting_factor


def load_excel_tower(path, toweronly=True):
    """Load tower properties from excel doc."""
    df = pd.read_excel(path, sheet_name='Tower Properties', usecols='B:K')
    if toweronly:
        df = df[df['Height [m]'] >= 15]
    return df


def calculate_mom_iner(out_diam, thick):
    """Calculate the moment of inertia."""
    r_out = out_diam / 2  # outer diameter [m]
    r_in = r_out - thick  # inner diameter [m]
    mom_iner = PI/4 * (r_out**4 - r_in**4)
    return mom_iner


def load_elastodyn_distprop(path):
    """Load distributed properties in ED file."""
    with open(path, 'r', encoding='utf-8') as f:
        for il, line in enumerate(f):
            if il == 3:
                ntwinpst = int(line.split()[0])
                break
    ed_distprop = np.loadtxt(path, skiprows=19, max_rows=ntwinpst)
    return ed_distprop


def load_yaml(path):
    """Load the yaml file."""
    with open(path, 'r', encoding='utf-8') as stream:
        try:
            yamldata = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yamldata


def load_hawc2_st(path):
    """Load HAWC2 structural file."""
    with open(path, 'r', encoding='utf-8') as f:
        for il, line in enumerate(f):
            if il == 2:
                ntwinpst = int(line.split()[1])
                break
    h2_st = np.loadtxt(path, skiprows=3, max_rows=ntwinpst)
    return h2_st


def normalize_tower_station(twr_stn):
    """Normalized twr stn to go from 0 to 1."""
    return (twr_stn - twr_stn[0]) / (twr_stn[-1] - twr_stn[0])


def get_body_length(htc_struc, body):
    """Get the length of a body from htc structure, given string name"""
    body_contents = htc_struc.get_subsection_by_name(body).c2_def.contents
    last_key = next(reversed(body_contents))
    length = abs(body_contents[last_key].values[-2])
    return length


def read_elastodyn_dat(path):
    """Get dictionary from an elastodyn dat file"""
    d = {}
    with open(path, 'r', encoding='utf-8') as ed:
        end = False
        for line in ed:
            contents = line.split()
            if contents[0] == 'OutList':
                end = True
            if end:
                break
            if not line.startswith('--'):
                try:
                    d[contents[1]] = float(contents[0])
                except ValueError:
                    d[contents[1]] = contents[0]
    return d

