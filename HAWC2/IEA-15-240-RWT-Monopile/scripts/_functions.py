"""Helper functions for making the tower/monopile models.

TODO: Move this to a top-level package so it can be reused in all subfolders and
also tests.
"""
import numpy as np
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

