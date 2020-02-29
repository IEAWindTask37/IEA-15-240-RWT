from __future__ import print_function

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import openmdao.api as om
from wisdem.rotorse.rotor import RotorSE, Init_RotorSE_wRefBlade
from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade
from wisdem.assemblies.fixed_bottom.monopile_assembly_turbine_nodrive import MonopileTurbine
from wisdem.aeroelasticse.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7

from generateTables import RWT_Tabular


# Global inputs and outputs
ontology_dir  = os.path.dirname( os.path.dirname( os.path.realpath(__file__)) ) + os.sep + 'WT_Ontology'
fname_schema  = ontology_dir + os.sep + 'IEAontology_schema.yaml'
fname_input   = ontology_dir + os.sep + 'IEA-15-240-RWT_FineGrid.yaml'
fname_output  = ontology_dir + os.sep + 'IEA-15-240-RWT_out.yaml'
folder_output = os.getcwd() + os.sep + 'outputs'

if not os.path.isdir(folder_output) and rank==0:
    os.mkdir(folder_output)


def initialize_problem(Analysis_Level):
    
    # Initialize blade design
    refBlade = ReferenceBlade()
    refBlade.verbose  = True
    refBlade.NINPUT       = 200
    Nsection_Tow          = 19
    refBlade.NPTS         = 200
    refBlade.spar_var     = ['Spar_cap_ss', 'Spar_cap_ps'] # SS, then PS
    refBlade.te_var       = 'TE_reinforcement'
    refBlade.validate     = False
    refBlade.fname_schema = fname_schema
    blade = refBlade.initialize(fname_input)
    
    FASTpref                        = {}
    FASTpref['Analysis_Level']      = Analysis_Level
    # Set FAST Inputs
    if Analysis_Level >= 1:
        # File management
        FASTpref['FAST_ver']            = 'OpenFAST'
        FASTpref['dev_branch']          = True
        FASTpref['FAST_exe']            = '~/local/bin/openfast'
        FASTpref['FAST_directory']      = '../OpenFAST'   # Path to fst directory files
        FASTpref['FAST_InputFile']      = 'IEA-15-240-RWT.fst' # FAST input file (ext=.fst)
        FASTpref['Turbsim_exe']         = '~/local/bin/turbsim'
        FASTpref['FAST_namingOut']      = 'IEA-15-240-RWT'
        FASTpref['FAST_runDirectory']   = 'temp/' + FASTpref['FAST_namingOut']
        
        # Run Settings
        FASTpref['cores']               = 1
        FASTpref['debug_level']         = 2 # verbosity: set to 0 for quiet, 1 & 2 for increasing levels of output

        # DLCs
        FASTpref['DLC_gust']            = None      # Max deflection
        # FASTpref['DLC_gust']            = RotorSE_DLC_1_4_Rated       # Max deflection    ### Not in place yet
        FASTpref['DLC_extrm']           = None      # Max strain
        # FASTpref['DLC_extrm']           = RotorSE_DLC_7_1_Steady      # Max strain        ### Not in place yet
        FASTpref['DLC_turbulent']       = None
        # FASTpref['DLC_turbulent']       = RotorSE_DLC_1_1_Turb      # Alternate turbulent case, replacing rated and extreme DLCs for calculating max deflection and strain
        FASTpref['DLC_powercurve']      = None      # AEP
        # FASTpref['DLC_powercurve']      = None      # AEP

        # Initialize, read initial FAST files to avoid doing it iteratively
        fast = InputReader_OpenFAST(FAST_ver=FASTpref['FAST_ver'], dev_branch=FASTpref['dev_branch'])
        fast.FAST_InputFile = FASTpref['FAST_InputFile']
        fast.FAST_directory = FASTpref['FAST_directory']
        fast.execute()
        fst_vt = fast.fst_vt
    else:
        fst_vt = {}

    prob = om.Problem()
    prob.model=MonopileTurbine(RefBlade=blade, Nsection_Tow=Nsection_Tow, VerbosityCosts=False, FASTpref=FASTpref)
    prob.model.nonlinear_solver = om.NonlinearRunOnce()
    prob.model.linear_solver    = om.DirectSolver()

    return prob, blade, fst_vt


def initialize_variables(prob, blade, Analysis_Level, fst_vt):

    # Initialize variable inputs
    prob.setup()

    prob = Init_RotorSE_wRefBlade(prob, blade, Analysis_Level = Analysis_Level, fst_vt = fst_vt)

    # Environmental parameters for the tower
    prob['significant_wave_height'] = 4.52
    prob['significant_wave_period'] = 9.45
    prob['water_depth']             = 30.
    prob['wind_reference_height'] = prob['hub_height'] = 150.
    prob['shearExp']                       = 0.11
    prob['rho']                            = 1.225
    prob['mu']                             = 1.7934e-5
    prob['water_density']                  = 1025.0
    prob['water_viscosity']                = 1.3351e-3
    prob['wind_beta'] = prob['wave_beta'] = 0.0
    prob['gust_stddev'] = 3

    # Steel properties for the tower
    prob['material_density']               = 7850.0
    prob['E']                              = 210e9
    prob['G']                              = 79.3e9
    prob['yield_stress']                   = 345e6
    prob['soil_G']                         = 140e6
    prob['soil_nu']                        = 0.4

    # Design constraints
    prob['max_taper_ratio']                = 0.4
    prob['min_diameter_thickness_ratio']   = 120.0

    # Safety factors
    prob['gamma_fatigue']   = 1.755 # (Float): safety factor for fatigue
    prob['gamma_f']         = 1.35  # (Float): safety factor for loads/stresses
    prob['gamma_m']         = 1.3   # (Float): safety factor for materials
    prob['gamma_freq']      = 1.1   # (Float): safety factor for resonant frequencies
    prob['gamma_n']         = 1.0
    prob['gamma_b']         = 1.1
    
    # Tower
    prob['tower_buckling_length']          = 30.0
    prob['tower_outfitting_factor']        = 1.07
    prob['foundation_height']       = -30.
    prob['suctionpile_depth']       = 45.
    prob['tower_section_height']    = np.array([5., 5., 5., 5., 5., 5., 5., 5., 5., 13.,  13.,  13.,  13.,  13.,  13.,  13.,  13.,  13., 12.58244309])
    prob['tower_outer_diameter'] = np.array([10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 9.92647687, 9.44319282, 8.83283769, 8.15148167, 7.38976138, 6.90908962, 6.74803581, 6.57231775, 6.5])
    prob['tower_wall_thickness'] = np.array([0.05534138, 0.05344902, 0.05150928, 0.04952705, 0.04751736, 0.04551709, 0.0435267, 0.04224176, 0.04105759, 0.0394965, 0.03645589, 0.03377851, 0.03219233, 0.03070819, 0.02910109, 0.02721289, 0.02400931, 0.0208264, 0.02399756])
    prob['tower_buckling_length']   = 15.0
    prob['transition_piece_mass']   = 100e3
    prob['transition_piece_height'] = 15.0


    prob['DC']      = 80.0
    prob['shear']   = True
    prob['geom']    = True
    prob['tower_force_discretization'] = 5.0
    prob['nM']      = 2
    prob['Mmethod'] = 1
    prob['lump']    = 0
    prob['tol']     = 1e-9
    prob['shift']   = 0.0
    
    # Offshore BOS
    prob['wtiv'] = 'example_wtiv'
    prob['feeder'] = 'future_feeder'
    prob['num_feeders'] = 1
    prob['oss_install_vessel'] = 'example_heavy_lift_vessel'
    prob['site_distance'] = 40.0
    prob['site_distance_to_landfall'] = 40.0
    prob['site_distance_to_interconnection'] = 40.0
    prob['plant_turbine_spacing'] = 7
    prob['plant_row_spacing'] = 7
    prob['plant_substation_distance'] = 1
    prob['tower_deck_space'] = 0.
    prob['nacelle_deck_space'] = 0.
    prob['blade_deck_space'] = 0.
    prob['port_cost_per_month'] = 2e6
    prob['monopile_deck_space'] = 0.
    prob['transition_piece_deck_space'] = 0.
    prob['commissioning_pct'] = 0.01
    prob['decommissioning_pct'] = 0.15
    prob['project_lifetime'] = prob['lifetime'] = 20.0    
    prob['number_of_turbines']             = 40
    prob['annual_opex']                    = 43.56 # $/kW/yr
    
    prob['tower_add_gravity'] = True

    # For turbine costs
    prob['offshore']             = True
    prob['crane']                = False
    prob['crane_cost']           = 0.0
    prob['labor_cost_rate']      = 3.0
    prob['material_cost_rate']   = 2.0
    prob['painting_cost_rate']   = 28.8
    
    # Drivetrain
    prob['bearing_number']          = 2
    prob['tilt']                    = 6.0
    prob['overhang']                = 10.99
    prob['hub_cm']                  = np.array([-10.604, 0.0, 5.462])
    prob['nac_cm']                  = np.array([-3.946, 0.0, 3.538])
    prob['hub_I']                   = np.array([1382171.187, 2169261.099, 2160636.794, 0.0, 0.0, 0.0])
    prob['nac_I']                   = np.array([7918328., 4751108., 5314813., 0.0, 0.0, 0.0])
    prob['hub_mass']                = 190e3
    prob['nac_mass']                = 6.309e5
    prob['hss_mass']                = 0.0
    prob['lss_mass']                = 15734.0
    prob['cover_mass']              = 0.0
    prob['pitch_system_mass']       = 0.0
    prob['platforms_mass']          = 11393 + 2*1973.
    prob['spinner_mass']            = 0.0
    prob['transformer_mass']        = 50e3
    prob['vs_electronics_mass']     = 0.0
    prob['yaw_mass']                = 100e3
    prob['gearbox_mass']            = 0.0
    prob['generator_mass']          = 226628.6 + 144963.1
    prob['bedplate_mass']           = 70328.7
    prob['main_bearing_mass']       = 5664

    return prob



def run_problem(prob):

    # Run initial condition no matter what
    print('Running at Initial Position:')
    prob.run_model()

    print('########################################')
    print('')
    print('Control variables')
    print('Rotor diam:    {:8.3f} m'.format(prob['diameter'][0]))
    print('TSR:           {:8.3f} -'.format(prob['control_tsr'][0]))
    print('Rated vel:     {:8.3f} m/s'.format(prob['rated_V'][0]))
    print('Rated rpm:     {:8.3f} rpm'.format(prob['rated_Omega'][0]))
    print('Rated pitch:   {:8.3f} deg'.format(prob['rated_pitch'][0]))
    print('Rated thrust:  {:8.3f} N'.format(prob['rated_T'][0]))
    print('Rated torque:  {:8.3f} N-m'.format(prob['rated_Q'][0]))
    print('')
    print('Constraints')
    print('Max TD:       {:8.3f} m'.format(prob['tip_deflection'][0]))
    print('TD ratio:     {:8.3f} -'.format(prob['tip_deflection_ratio'][0]))
    print('Blade root M: {:8.3f} N-m'.format(prob['root_bending_moment'][0]))
    print('')
    print('Objectives')
    print('AEP:         {:8.3f} GWh'.format(prob['AEP'][0]))
    print('LCoE:        {:8.4f} $/MWh'.format(prob['lcoe'][0]))
    print('')
    print('Blades')
    print('Blade mass:  {:8.3f} kg'.format(prob['mass_one_blade'][0]))
    print('Blade cost:  {:8.3f} $'.format(prob['total_blade_cost'][0]))
    print('Blade freq:  {:8.3f} Hz'.format(prob['freq_curvefem'][0]))
    print('3 blade M_of_I:  ', prob['I_all_blades'], ' kg-m^2')
    print('Hub M:  ', prob['Mxyz_total'], ' kg-m^2')
    print('')
    print('RNA Summary')
    print('RNA mass:    {:8.3f} kg'.format(prob['tow.pre.mass'][0]))
    print('RNA C_of_G (TT):  ', prob['rna_cg'], ' m')
    print('RNA M_of_I:  ', prob['tow.pre.mI'], ' kg-m^2')
    print('')
    print('Tower')
    print('Tower top F: ', prob['tow.pre.rna_F'], ' N')
    print('Tower top M: ', prob['tow.pre.rna_M'], ' N-m')
    print('Tower freqs: ', prob['tow.post.structural_frequencies'], ' Hz')
    print('Tower vel:   {:8.3f} kg'.format(prob['tow.wind.Uref'][0]))
    print('Tower mass:  {:8.3f} kg'.format(prob['tower_mass'][0]))
    print('Tower cost:  {:8.3f} $'.format(prob['tower_cost'][0]))
    print('Monopile mass:  {:8.3f} kg'.format(prob['monopile_mass'][0]))
    print('Monopile cost:  {:8.3f} $'.format(prob['monopile_cost'][0]))
    print('########################################')

    # Complete data dump
    #prob.model.list_inputs(units=True)
    #prob.model.list_outputs(units=True)
    return prob


def postprocess(prob, blade):

    def format_save(fig, fig_name):
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig.savefig(folder_output + os.sep + fig_name+'.pdf', pad_inches=0.1, bbox_inches='tight')
        fig.savefig(folder_output + os.sep + fig_name+'.png', pad_inches=0.1, bbox_inches='tight')
        
    # Problem initialization
    var_y           = ['chord','theta','rthick','p_le','precurve','presweep']
    label_y         = ['Chord [m]', 'Twist [deg]', 'Relative Thickness [%]', 'Pitch Axis Chord Location [%]', 'Prebend [m]', 'Sweep [m]']
    scaling_factor  = [1. , 1. , 100. , 100., 1., 1.]

    figsize=(5.3, 4)
    fig = plt.figure(figsize=figsize)
    for i in range(len(var_y)):
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(blade['pf']['r'], blade['pf'][var_y[i]] * scaling_factor[i], 'k', linewidth=2)
        plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
        plt.ylabel(label_y[i], fontsize=14, fontweight='bold')
        fig_name = var_y[i] + '_dimensional'
        format_save(fig, fig_name)

        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(blade['pf']['s'], blade['pf'][var_y[i]] * scaling_factor[i], 'k', linewidth=2)
        plt.xlabel('Nondimensional Blade Span (r/R)', fontsize=14, fontweight='bold')
        plt.ylabel(label_y[i], fontsize=14, fontweight='bold')
        fig_name = var_y[i] + '_nondimensional'
        format_save(fig, fig_name)

    # Pitch
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(prob['V'], prob['pitch'], linewidth=2)
    plt.xlabel('Wind [m/s]', fontsize=14, fontweight='bold')
    plt.ylabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
    fig_name = 'pitch'
    format_save(fig, fig_name)

    # Power curve
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(prob['V'], prob['P'] * 1.00e-006, linewidth=2)
    plt.xlabel('Wind [m/s]', fontsize=14, fontweight='bold')
    plt.ylabel('Electrical Power [MW]', fontsize=14, fontweight='bold')
    plt.yticks(np.arange(16))
    fig_name = 'power'
    format_save(fig, fig_name)

    # ELEC Coefficient of Power curve
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(prob['V'], prob['Cp'], linewidth=2)
    plt.xlabel('Wind [m/s]', fontsize=14, fontweight='bold')
    plt.ylabel('Power Coefficient', fontsize=14, fontweight='bold')
    plt.yticks(1e-2*np.arange(0, 51, 5))
    fig_name = 'coefficient_power'
    format_save(fig, fig_name)

    # AERO Coefficient of Power curve
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(prob['V'], prob['Cp_aero'], linewidth=2)
    plt.xlabel('Wind [m/s]', fontsize=14, fontweight='bold')
    plt.ylabel('Power Coefficient', fontsize=14, fontweight='bold')
    plt.yticks(1e-2*np.arange(0, 51, 5))
    fig_name = 'coefficient_power_aero'
    format_save(fig, fig_name)

    # Omega
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(prob['V'], prob['Omega'], linewidth=2)
    plt.xlabel('Wind [m/s]', fontsize=14, fontweight='bold')
    plt.ylabel('Rotor Speed [rpm]', fontsize=14, fontweight='bold')
    fig_name = 'omega'
    format_save(fig, fig_name)

    # Tip speed
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(prob['V'], prob['Omega'] * np.pi / 30. * prob['r'][-1], linewidth=2)
    plt.xlabel('Wind [m/s]', fontsize=14, fontweight='bold')
    plt.ylabel('Blade Tip Speed [m/s]', fontsize=14, fontweight='bold')
    fig_name = 'tip_speed'
    format_save(fig, fig_name)

    # Thrust
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(prob['V'], prob['T'] * 1.00e-006, linewidth=2)
    plt.xlabel('Wind [m/s]', fontsize=14, fontweight='bold')
    plt.ylabel('Rotor Thrust [MN]', fontsize=14, fontweight='bold')
    fig_name = 'thrust'
    format_save(fig, fig_name)

    # Coefficient Thrust
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(prob['V'], prob['Ct_aero'], linewidth=2)
    plt.xlabel('Wind [m/s]', fontsize=14, fontweight='bold')
    plt.ylabel('Thrust Coefficient', fontsize=14, fontweight='bold')
    plt.yticks(1e-1*np.arange(0, 8.1))
    fig_name = 'coefficient_thrust'
    format_save(fig, fig_name)

    # Torque
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(prob['V'], prob['Q'] * 1.00e-006, linewidth=2)
    plt.xlabel('Wind [m/s]', fontsize=14, fontweight='bold')
    plt.ylabel('Rotor Torque [MNm]', fontsize=14, fontweight='bold')
    fig_name = 'torque'
    format_save(fig, fig_name)

    # Tabular output: Blade
    temp = np.c_[blade['pf']['s'], blade['pf']['r']]
    for iy,y in enumerate(var_y):
        temp = np.c_[temp, blade['pf'][y]*scaling_factor[iy]]
    bladeDF = pd.DataFrame(data=temp, columns=['Blade Span','Rotor Coordinate [m]'] + label_y)
    
    # Tabular output: Rotor Performance
    perfDF = pd.DataFrame(data=np.c_[prob['V'],prob['pitch'], prob['P']*1e-6, prob['Omega'], prob['Omega']*prob['r'][-1]*np.pi/30., prob['T']*1e-6, prob['Q']*1e-6],
                         columns=['Wind [m/s]','Pitch [deg]','Power [MW]','Rotor Speed [rpm]','Tip Speed [m/s]','Thrust [MN]','Torque [MNm]'])

    # Tabular output: Tower 
    htow = np.cumsum(np.r_[0.0, prob['suctionpile_depth'], prob['tower_section_height']]) - (prob['water_depth']+prob['suctionpile_depth'])
    towdata = np.c_[htow,
                    np.r_[prob['tower_outer_diameter'][0], prob['tower_outer_diameter']],
                    np.r_[prob['tower_wall_thickness'][0], prob['tower_wall_thickness'][0], prob['tower_wall_thickness']]]
    rowadd = []
    for k in range(towdata.shape[0]):
        if k==0: continue
        if k+1 < towdata.shape[0]:
            rowadd.append([towdata[k,0]+1e-3, towdata[k,1], towdata[k+1,2]])
    towdata = np.vstack((towdata, rowadd))
    towdata[:,-1] *= 1e3
    towdata = np.round( towdata[towdata[:,0].argsort(),], 3)
    colstr = ['Height [m]','OD [m]', 'Thickness [mm]']
    towDF = pd.DataFrame(data=towdata, columns=colstr)
    mycomments = ['']*towdata.shape[0]
    mycomments[0] = 'Monopile start'
    mycomments[np.where(towdata[:,0] == -prob['water_depth'])[0][0]] = 'Mud line'
    mycomments[np.where(towdata[:,0] == 0.0)[0][0]] = 'Water line'
    mycomments[np.where(towdata[:,0] == prob['transition_piece_height'])[0][0]] = 'Tower start'
    mycomments[-1] = 'Tower top'
    towDF['Location'] = mycomments
    towDF = towDF[['Location']+colstr]
    A = 0.25*np.pi*(towDF['OD [m]']**2 - (towDF['OD [m]']-2*1e-3*towDF['Thickness [mm]'])**2)
    I = (1/64.)*np.pi*(towDF['OD [m]']**4 - (towDF['OD [m]']-2*1e-3*towDF['Thickness [mm]'])**4)
    towDF['Mass Density [kg/m]'] = 7850 * A
    towDF['Fore-aft inertia [kg.m]'] = towDF['Mass Density [kg/m]'] * I/A
    towDF['Side-side inertia [kg.m]'] = towDF['Mass Density [kg/m]'] * I/A
    towDF['Fore-aft stiffness [N.m^2]'] = 2e11 * I
    towDF['Side-side stiffness [N.m^2]'] = 2e11 * I
    towDF['Torsional stiffness [N.m^2]'] = 7.93e10 * 2*I
    towDF['Axial stiffness [N]'] = 2e11 * A
    with open('tow.tbl','w') as f:
        towDF.to_latex(f, index=False)

    # Tower stiffness plots
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(towDF['Height [m]'], towDF['Mass Density [kg/m]'], linewidth=2)
    plt.xlabel('Location [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Mass Density [kg/m]', fontsize=14, fontweight='bold')
    fig_name = 'tower_massdens'
    format_save(fig, fig_name)

    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(towDF['Height [m]'], towDF['Fore-aft inertia [kg.m]'], linewidth=2)
    plt.xlabel('Location [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Fore-aft/side-side inertia [kg.m]', fontsize=14, fontweight='bold')
    fig_name = 'tower_foreaft_sideside-inertia'
    format_save(fig, fig_name)

    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(towDF['Height [m]'], towDF['Fore-aft stiffness [N.m^2]'], linewidth=2)
    ax.plot(towDF['Height [m]'], towDF['Torsional stiffness [N.m^2]'], linewidth=2)
    ax.legend(('Fore-aft/side-side','Torsional'), loc='best')
    plt.xlabel('Location [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Stiffness [N.m^2]', fontsize=14, fontweight='bold')
    fig_name = 'tower_stiffness'
    format_save(fig, fig_name)

    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(towDF['Height [m]'], towDF['Axial stiffness [N]'], linewidth=2)
    plt.xlabel('Location [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Axial Stiffness [N]', fontsize=14, fontweight='bold')
    fig_name = 'tower_axial_stiffness'
    format_save(fig, fig_name)

    # Blade stiffness plots- superceded by SONATA/VABS 6x6 outputs
    bladeStiff   = np.c_[prob['z'], prob['EA'], prob['EIxx'], prob['EIyy'], prob['EIxy'], prob['GJ'], prob['rhoA'], prob['rhoJ'], 1e3*prob['x_ec'], 1e3*prob['y_ec']]
    bladeStiffDF = pd.DataFrame(data=bladeStiff, columns=['Blade z-coordinate [m]',
                                                          'Axial stiffness [N]',
                                                          'Edgewise stiffness [Nm^2]',
                                                          'Flapwise stiffness [Nm^2]',
                                                          'Flap-edge coupled stiffness [Nm^2]',
                                                          'Torsional stiffness [Nm^2]',
                                                          'Mass density [kg/m]',
                                                          'Polar moment of intertia density [kg/m]',
                                                          'X-distance to elastic center [mm]',
                                                          'Y-distance to elastic center [mm]'])

    xx = prob['z'] / prob['z'].max()
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(xx, prob['rhoA'], linewidth=2)
    plt.xlabel('Nondimensional Blade Span (r/R)', fontsize=14, fontweight='bold')
    plt.ylabel('Mass density [kg/m]', fontsize=14, fontweight='bold')
    fig_name = 'blade_mass'
    format_save(fig, fig_name)

    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(xx, prob['EIxx'], linewidth=2)
    ax.plot(xx, prob['EIyy'], linewidth=2)
    ax.plot(xx, prob['GJ'], linewidth=2)
    ax.legend(('Edge','Flap','Torsional'), loc='best')
    plt.xlabel('Nondimensional Blade Span (r/R)', fontsize=14, fontweight='bold')
    plt.ylabel('Stiffness [N.m^2]', fontsize=14, fontweight='bold')
    fig_name = 'blade_stiffness'
    format_save(fig, fig_name)

    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(xx, prob['EA'], linewidth=2)
    plt.xlabel('Nondimensional Blade Span (r/R)', fontsize=14, fontweight='bold')
    plt.ylabel('Axial Stiffness [N]', fontsize=14, fontweight='bold')
    fig_name = 'blade_axial_stiffness'
    format_save(fig, fig_name)
    
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(xx, 1e3*prob['x_ec'], linewidth=2)
    ax.plot(xx, 1e3*prob['y_ec'], linewidth=2)
    ax.legend(('x-distance','y-distance'), loc='best')
    plt.xlabel('Nondimensional Blade Span (r/R)', fontsize=14, fontweight='bold')
    plt.ylabel('Distance to elastic center [mm]', fontsize=14, fontweight='bold')
    fig_name = 'blade_ec'
    format_save(fig, fig_name)
    
    # Tower plot- geometry
    brown = np.array([150., 75., 0.])/256.
    #fig, ax = plt.subplots(1,1,figsize=(11,4))
    fig = plt.figure(figsize=(11,4))
    ax1 = fig.add_subplot(121)
    lab1 = ax1.plot(towdata[:,1], towdata[:,0], 'k', linewidth=2)
    vx = ax1.get_xlim()
    lab2 = ax1.plot(vx, np.zeros(2), color='b', linestyle='--')
    lab3 = ax1.plot(vx, -prob['water_depth']*np.ones(2), color=brown, linestyle='--')
    lab4 = ax1.plot(vx, prob['transition_piece_height']*np.ones(2), color='g', linestyle='--')
    ax1.text(vx[0]+0.02*np.diff(vx), 2, 'Water line', color='b', fontsize=12)
    ax1.text(vx[0]+0.02*np.diff(vx), -prob['water_depth']+2, 'Mud line', color=brown, fontsize=12)
    ax1.text(vx[0]+0.02*np.diff(vx), prob['transition_piece_height']+2, 'Tower transition', color='g', fontsize=12)
    ax1.set_xlim(vx)
    plt.xlabel('Outer Diameter [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Tower Height [m]', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(color=[0.8,0.8,0.8], linestyle='--')
    #fig_name = 'tower_diameter'
    #fig.savefig(folder_output + os.sep + fig_name, pad_inches=0.1, bbox_inches='tight')

    #fig.clf()
    #ax = fig.add_subplot(111)
    ax2 = fig.add_subplot(122)
    lab1 = ax2.plot(towdata[:,2], towdata[:,0], 'k', linewidth=2)
    vx = ax2.get_xlim()
    lab2 = ax2.plot(vx, np.zeros(2), color='b', linestyle='--')
    lab3 = ax2.plot(vx, -prob['water_depth']*np.ones(2), color=brown, linestyle='--')
    lab4 = ax2.plot(vx, 20*np.ones(2), color='g', linestyle='--')
    ax2.set_xlim(vx)
    plt.xlabel('Wall Thickness [mm]', fontsize=14, fontweight='bold')
    #plt.ylabel('Tower Height [m]', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    #plt.yticks(fontsize=12)
    plt.setp( ax2.get_yticklabels(), visible=False)
    plt.grid(color=[0.8,0.8,0.8], linestyle='--')
    plt.subplots_adjust(bottom = 0.15, left = 0.15)
    #fig_name = 'tower_thickness'
    fig_name = 'tower_geometry'
    fig.subplots_adjust(hspace=0.02, wspace=0.02, bottom = 0.15, left = 0.15)
    fig.savefig(folder_output + os.sep + fig_name+'.pdf', pad_inches=0.1, bbox_inches='tight')
    fig.savefig(folder_output + os.sep + fig_name+'.png', pad_inches=0.1, bbox_inches='tight')

    # Frequency plot
    fn_tower = [prob['tow.tower.f1'], prob['tow.tower.f2']]
    f        = np.linspace(0., 0.5, num=1000)[1:]
    omega    = f*(2.*np.pi)

    Omega_rot_min = prob['control_minOmega']/60. #5./60
    Omega_rot_max = prob['control_maxOmega']/60. #7.56/60

    f_1P = [0, Omega_rot_min, Omega_rot_min, Omega_rot_max, Omega_rot_max, f[-1]]
    f_3P = [0, 3.*Omega_rot_min, 3.*Omega_rot_min, 3.*Omega_rot_max, 3.*Omega_rot_max, f[-1]]
    NP_y = [0., 0., 1., 1., 0., 0.]

    # f_1P_marg_low  = Omega_rot_min/1.1
    # f_1P_marg_high = Omega_rot_min*1.1
    # f_3P_marg_low  = Omega_rot_min*3./1.1
    # f_3P_marg_high = Omega_rot_min*3.*1.1

    f_tower = [fn_tower[0], fn_tower[0]]
    y_tower = [0., 1.]

    # Kaimal
    Sigma1 = 42
    V_hub  = 10.
    L_k = 8.1*Sigma1
    Sk = np.zeros_like(f)
    for i, fi in enumerate(f):
        Sk[i] = 4*fi*L_k/V_hub/(1+6*fi*L_k/V_hub)**(5./3.)/fi
    Sk /= max(Sk)

    # Pierson-Moskowitz
    F = 500000
    g = 9.81
    U10 = 10.

    U19     = U10*1.17
    alpha   = 8.1e-3
    Beta    = 0.74
    omega_0 = g/U19
    omega_p = 0.877*g/U19

    S = np.zeros_like(f)
    for i, omega_i in enumerate(omega):
        S[i] = alpha*g**2./omega_i**5.*np.exp(-Beta*(omega_0/omega_i)**4.)
    S /= max(S)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6., 2.5))
    fig.subplots_adjust(bottom=0.36, top=0.975,  left=0.12, right=0.975)#, right=0.8, hspace=0.3, wspace=0.3)

    ax.fill(f_1P, NP_y, color=[0.5,0.5,0.5], label='1P')
    ax.fill(f_3P, NP_y, color=[0.75,0.75,0.75], label='3P')
    ax.plot(f, Sk, color='r', label='Wind,\nKaimal Spect.', linewidth=1.5)
    ax.plot(f, S,  color='b', label="Waves,\nJONSWAP Spect.", linewidth=1.5)
    ax.plot(f_tower, y_tower, color='k', label="Tower,\n1st Nat. Freq.", linewidth=1.5)
    ax.grid(color=[0.8,0.8,0.8], linestyle='--')
    ax.set_ylabel('Normalized PSD')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_xlim((0.,max(f)))

    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=5, prop={'size':9})
    fig_name = 'tower_fn'
    fig.savefig(folder_output + os.sep + fig_name+'.pdf', pad_inches=0.1, bbox_inches='tight')
    fig.savefig(folder_output + os.sep + fig_name+'.png', pad_inches=0.1, bbox_inches='tight')


    # Write all tabular data to xlsx
    myobj = RWT_Tabular(fname_input, towDF=towDF, rotDF=perfDF)
    myobj.write_all()


if __name__ == "__main__":
    # Initialize output container
    if not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    # Set FAST integration
    # 0: Run CCBlade; 1: Update FAST model at each iteration but do not run; 2: Run FAST w/ ElastoDyn; 3: (Not implemented) Run FAST w/ BeamDyn
    Analysis_Level = 0

    # Seed inputs
    prob, blade, fst_vt = initialize_problem(Analysis_Level)
    prob = initialize_variables(prob, blade, Analysis_Level, fst_vt)

    # Run the analysis
    prob = run_problem(prob)    

    # Generate output plots, tables, and Excel sheet
    postprocess(prob, blade)
