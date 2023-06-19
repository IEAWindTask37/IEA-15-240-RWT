# -*- coding: utf-8 -*-
"""Check that basic RNA props in the HAWC2-OpenFAST onshore models match.

Requires wetb, numpy.

TODO: Add checks for monopile and umaine models
"""
import numpy as np
from wetb.hawc2 import HTCFile

import _test_functions as tstf
from _test_functions import FROOT


def test_of_h2_onshore():
    """Check RNA properties in OF Monopile model versus H2 onshore, UMaine, monopile
    """

    ed_path = FROOT / 'OpenFAST'/  'IEA-15-240-RWT-Monopile'/  'IEA-15-240-RWT-Monopile_ElastoDyn.dat'
    h2_dir = FROOT/  'HAWC2'/  'IEA-15-240-RWT-Onshore'
    h2_path = h2_dir/  'htc'/  'IEA_15MW_RWT_Onshore.htc'

    h2_dir = h2_dir.as_posix()  # wetb requires strings, not Path objects...
    h2_path = h2_path.as_posix()
    
    ed_dict = tstf.read_elastodyn_dat(ed_path)
    htc = HTCFile(h2_path, modelpath=h2_dir)
    
    htc_struc = htc.new_htc_structure
    
    # tower
    z_towerbottom = -htc_struc.orientation.base.inipos[2]  # location of tower bottom in space
    twrht_h2 = tstf.get_body_length(htc_struc, 'tower') + z_towerbottom
    assert np.isclose(ed_dict['TowerHt'], twrht_h2)  # tower height
        
    # nacelle and yaw bearing masses and inertias
    assert np.isclose(ed_dict['YawBrMass'], htc_struc.get_subsection_by_name('towertop').concentrated_mass__1.values[4]) # yaw bearing mass
    assert np.isclose(ed_dict['NacCMxn'], htc_struc.get_subsection_by_name('towertop').concentrated_mass__2.values[2], atol=1e-3)  # nacelle cm
    assert np.isclose(ed_dict['NacCMzn'], -htc_struc.get_subsection_by_name('towertop').concentrated_mass__2.values[3], atol=1e-3)  # nacelle cm
    assert np.isclose(ed_dict['NacMass'], htc_struc.get_subsection_by_name('towertop').concentrated_mass__2[4])  # nacelle mass
    
    # generator and hub inertia
    assert np.isclose(ed_dict['GenIner'], htc_struc.get_subsection_by_name('shaft').concentrated_mass__1.values[-1])  # generator inertia
    assert np.isclose(ed_dict['HubMass'], htc_struc.get_subsection_by_name('shaft').concentrated_mass__2.values[4])  # hub mass
    assert np.isclose(ed_dict['HubIner'], htc_struc.get_subsection_by_name('shaft').concentrated_mass__2.values[-1])  # hub inertia
    
    # hub radius, shaft tilt and coning
    assert np.isclose(ed_dict['HubRad'], htc_struc.get_subsection_by_name('hub1').c2_def.sec__2.values[-2])  # hub radius
    assert np.isclose(-ed_dict['ShftTilt'], htc_struc.orientation.relative__2.mbdy2_eulerang__2.values[0])  # tilt
    
    # hub height
    tilt = 6 * np.pi / 180
    z_hub = 150
    ttop_length = tstf.get_body_length(htc_struc, 'towertop')
    conn_length = tstf.get_body_length(htc_struc, 'connector')
    shaft_length = tstf.get_body_length(htc_struc, 'shaft')
    z_hub_h2 = twrht_h2 + ttop_length + conn_length*np.sin(tilt) + shaft_length*np.sin(tilt)
    z_hub_of = ed_dict['TowerHt'] + ed_dict['Twr2Shft'] + -ed_dict['OverHang']*np.tan(tilt)
    assert np.isclose(z_hub_h2, z_hub, atol=1e-2)
    assert np.isclose(z_hub_of, z_hub, atol=1e-2)

    # overhang distance (measured along tilted axis)
    overhang_h2 = shaft_length + conn_length
    overhang_of = -ed_dict['OverHang']
    assert overhang_h2 == overhang_of
