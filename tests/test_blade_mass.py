from util.FAST_reader import InputReader_OpenFAST
import numpy as np
import os

# Paths to files

    
local_dir = os.path.dirname( os.path.realpath(__file__) )
BDtw_path = os.path.join(local_dir, '../OpenFAST/IEA-15-240-RWT')
BDc2_path = os.path.join(local_dir, '../HAWC2/converted/')
h2_path_FPM = os.path.join(local_dir, '../HAWC2/IEA-15-240-RWT-FixedBottom/data/blade_beamdyn_c2_FPM.st')
h2_path_noFPM = os.path.join(local_dir, '../HAWC2/IEA-15-240-RWT-FixedBottom/data/blade_beamdyn_c2_noFPM.st')

base_name1    = 'IEA-15-240-RWT'
blade_length = 117.2
ref_blade_mass = 69. * 1e+3

def test_blade_mass_BDtw():

    # Check mass beamdyn defined along twist centers
    readBD1 = InputReader_OpenFAST()
    readBD1.FAST_directory = BDtw_path
    readBD1.fst_vt = {}
    readBD1.fst_vt['BeamDyn'] = {}
    readBD1.fst_vt['BeamDynBlade'] = {}
    readBD1.fst_vt['Fst'] = {}
    readBD1.fst_vt['outlist'] = {}
    readBD1.fst_vt['outlist']['BeamDyn'] = {}
    readBD1.fst_vt['Fst']['BDBldFile(1)'] = base_name1 + '_BeamDyn.dat'
    bd_file = os.path.join(BDtw_path, readBD1.fst_vt['Fst']['BDBldFile(1)'])
    readBD1.read_BeamDyn(bd_file)
    BDtw = np.trapz(readBD1.fst_vt['BeamDynBlade']['beam_inertia'][:,0,0], readBD1.fst_vt['BeamDynBlade']['radial_stations']*blade_length)
    assert np.isclose(BDtw,ref_blade_mass)

def test_blade_mass_BDc2():

    # Check mass beamdyn defined along mid chord
    readBD2 = InputReader_OpenFAST()
    readBD2.FAST_directory = BDc2_path
    readBD2.fst_vt = {}
    readBD2.fst_vt['BeamDyn'] = {}
    readBD2.fst_vt['BeamDynBlade'] = {}
    readBD2.fst_vt['Fst'] = {}
    readBD2.fst_vt['outlist'] = {}
    readBD2.fst_vt['outlist']['BeamDyn'] = {}
    readBD2.fst_vt['Fst']['BDBldFile(1)'] = base_name1 + '_BeamDyn_c2.dat'
    bd_file2 = os.path.join(BDc2_path, readBD2.fst_vt['Fst']['BDBldFile(1)'])
    readBD2.read_BeamDyn(bd_file2)
    BDc2 = np.trapz(readBD2.fst_vt['BeamDynBlade']['beam_inertia'][:,0,0], readBD2.fst_vt['BeamDynBlade']['radial_stations']*blade_length)
    assert np.isclose(BDc2,ref_blade_mass)

def test_blade_mass_ED():
    # Load elastodyn blade
    readED = InputReader_OpenFAST()
    readED.FAST_directory = BDtw_path
    readED.fst_vt['ElastoDyn'] = {}
    readED.fst_vt['ElastoDynBlade'] = {}
    readED.fst_vt['ElastoDyn']['BldFile1'] = base_name1 + '_ElastoDyn_blade.dat'
    ed_file = os.path.join(BDtw_path, readED.fst_vt['ElastoDyn']['BldFile1'])
    readED.read_ElastoDynBlade(ed_file)
    ED = np.trapz(readED.fst_vt['ElastoDynBlade']['BMassDen'], np.array(readED.fst_vt['ElastoDynBlade']['BlFract'])*blade_length)
    assert np.isclose(ED,ref_blade_mass)

def test_blade_mass_H2():

    # Load H2 FPM and no FPM
    h2FPM = np.loadtxt(h2_path_FPM, skiprows = 5)
    h2noFPM = np.loadtxt(h2_path_noFPM, skiprows = 5)
    H2FPM = np.trapz(h2FPM[:,1], h2FPM[:,0])
    H2noFPM = np.trapz(h2noFPM[:,1], h2noFPM[:,0])

    assert np.isclose(H2FPM,ref_blade_mass)
    assert np.isclose(H2noFPM,ref_blade_mass)