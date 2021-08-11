# -*- coding: utf-8 -*-
"""Run basic test to compare values from OpenFAST files with HAWC2.

Requires wetb, numpy
"""
import numpy as np
from wetb.hawc2 import HTCFile


def read_elastodyn_dat(path):
    """Get dictionary from an elastodyn dat file"""
    d = {}
    with open(path, 'r') as ed:
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
    

def test_openfast_hawc2_match():
    """compare a series of values in the monopile elastodyn and fixed-bottom hawc2 models"""

    ed_path = './OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_ElastoDyn.dat'
    h2_path = './HAWC2/IEA-15-240-RWT-FixedBottom/htc/IEA_15MW_RWT.htc'
    
    ed_dict = read_elastodyn_dat(ed_path)
    htc = HTCFile(h2_path)
    
    htc_struc = htc.new_htc_structure
    
    # tower
    assert np.isclose(ed_dict['TowerHt'], -htc_struc.get_subsection_by_name('tower').c2_def.sec__11.values[-2])  # tower height
    
    # nacelle and yaw bearing masses and inertias
    assert np.isclose(ed_dict['YawBrMass'], htc_struc.get_subsection_by_name('towertop').concentrated_mass__1.values[4]) # yaw bearing mass
    assert np.isclose(ed_dict['NacCMxn'], htc_struc.get_subsection_by_name('towertop').concentrated_mass__2.values[2])  # nacelle cm
    assert np.isclose(ed_dict['NacCMzn'], -htc_struc.get_subsection_by_name('towertop').concentrated_mass__2.values[3])  # nacelle cm
    assert np.isclose(ed_dict['NacMass'], htc_struc.get_subsection_by_name('towertop').concentrated_mass__2[4])  # nacelle mass
    
    # generator and hub inertia
    assert np.isclose(ed_dict['GenIner'], htc_struc.get_subsection_by_name('shaft').concentrated_mass__1.values[-1])  # generator inertia
    assert np.isclose(ed_dict['HubMass'], htc_struc.get_subsection_by_name('shaft').concentrated_mass__2.values[4])  # hub mass
    assert np.isclose(ed_dict['HubIner'], htc_struc.get_subsection_by_name('shaft').concentrated_mass__2.values[-1])  # hub inertia
    
    # hub radius, shaft tilt and coning
    assert np.isclose(ed_dict['HubRad'], htc_struc.get_subsection_by_name('hub1').c2_def.sec__2.values[-2])  # hub radius
    assert np.isclose(-ed_dict['ShftTilt'], htc_struc.orientation.relative__2.body2_eulerang__2.values[0])  # tilt

