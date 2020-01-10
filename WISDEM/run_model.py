from __future__ import print_function

import os
import shutil

import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import pandas as pd

import openmdao.api as om
from wisdem.rotorse.rotor import RotorSE, Init_RotorSE_wRefBlade
from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade
from wisdem.commonse.turbine_constraints import TurbineConstraints
from wisdem.assemblies.fixed_bottom.monopile_assembly_turbine_nodrive import MonopileTurbine, Init_MonopileTurbine
from wisdem.aeroelasticse.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7

from generateTables import RWT_Tabular


# Global inputs and outputs
fname_schema  = 'IEAontology_schema.yaml'
fname_input   = 'IEA-15-240-RWT.yaml'
fname_output  = 'IEA-15-240-RWT_out.yaml'
folder_output = os.getcwd() + os.sep + 'outputs'


# Class to print outputs on screen
class Outputs_2_Screen(om.ExplicitComponent):
    def setup(self):
        # self.add_input('chord',                val=np.zeros(NPTS))
        # self.add_input('theta',                val=np.zeros(NPTS))
        self.add_input('bladeLength',          val=0.0, units = 'm')
        self.add_input('total_blade_cost',     val=0.0, units = 'USD')
        self.add_input('mass_one_blade',       val=0.0, units = 'kg')
        self.add_input('tower_mass',           val=0.0, units = 'kg')
        self.add_input('tower_cost',           val=0.0, units = 'USD')
        self.add_input('control_tsr',          val=0.0)
        self.add_input('AEP',                  val=0.0, units = 'GW * h')
        self.add_input('lcoe',                 val=0.0, units = 'USD/MW/h')
        self.add_input('rated_T',              val=0.0, units = 'MN')
        self.add_input('root_bending_moment',  val=0.0, units = 'MN * m')
        self.add_input('tip_deflection',       val=0.0, units = 'm')
        self.add_input('tip_deflection_ratio', val=0.0)

        
    def compute(self, inputs, outputs):
        print('########################################')
        print('Optimization variables')
        # print('Max chord:   {:8.3f} m'.format(max(inputs['chord'])))
        print('TSR:         {:8.3f} -'.format(inputs['control_tsr'][0]))
        print('')
        print('Constraints')
        print('Max TD:      {:8.3f} m'.format(inputs['tip_deflection'][0]))
        print('TD ratio:    {:8.10f} -'.format(inputs['tip_deflection_ratio'][0]))
        print('')
        print('Objectives')
        print('AEP:         {:8.10f} GWh'.format(inputs['AEP'][0]))
        print('Blade mass:  {:8.3f} kg'.format(inputs['mass_one_blade'][0]))
        print('Blade cost:  {:8.3f} $'.format(inputs['total_blade_cost'][0]))
        print('Tower mass:  {:8.3f} kg'.format(inputs['tower_mass'][0]))
        print('Tower cost:  {:8.3f} $'.format(inputs['tower_cost'][0]))
        print('LCoE:        {:8.3f} $/MWh'.format(inputs['lcoe'][0]))
        print('########################################')

        
class Convergence_Trends_Opt(om.ExplicitComponent):
    def initialize(self):
        
        self.options.declare('folder_output')
        self.options.declare('optimization_log')
        
    def compute(self, inputs, outputs):
        
        folder_output       = self.options['folder_output']
        optimization_log    = self.options['folder_output'] + self.options['optimization_log']

        if os.path.exists(optimization_log):
        
            cr = CaseReader(optimization_log)
            cases = cr.list_cases()
            rec_data = {}
            iterations = []
            for i, casei in enumerate(cases):
                iterations.append(i)
                it_data = cr.get_case(casei)
                
                # parameters = it_data.get_responses()
                for parameters in [it_data.get_responses(), it_data.get_design_vars()]:
                    for j, param in enumerate(parameters.keys()):
                        if i == 0:
                            rec_data[param] = []
                        rec_data[param].append(parameters[param])

            for param in rec_data.keys():
                fig, ax = plt.subplots(1,1,figsize=(5.3, 4))
                ax.plot(iterations, rec_data[param])
                ax.set(xlabel='Number of Iterations' , ylabel=param)
                fig_name = 'Convergence_trend_' + param + '.pdf'
                fig.savefig(folder_output + fig_name)
                plt.close(fig)

                
# Group to link the openmdao components
class Optimize_MonopileTurbine(om.Group):

    def initialize(self):
        self.options.declare('RefBlade')
        self.options.declare('folder_output' ,  default='')
        self.options.declare('FASTpref',        default={})
        self.options.declare('Nsection_Tow',    default = 19)
        self.options.declare('VerbosityCosts',  default = False)
        self.options.declare('user_update_routine',     default=None)
        
    def setup(self):
        RefBlade             = self.options['RefBlade']
        folder_output        = self.options['folder_output']
        Nsection_Tow         = self.options['Nsection_Tow']
        VerbosityCosts       = self.options['VerbosityCosts']
        user_update_routine  = self.options['user_update_routine']
        FASTpref             = self.options['FASTpref']
    
        self.add_subsystem('lb_wt', MonopileTurbine(RefBlade=RefBlade, Nsection_Tow = Nsection_Tow, VerbosityCosts = VerbosityCosts, user_update_routine=user_update_routine, FASTpref=FASTpref), promotes=['*'])
        # Post-processing
        self.add_subsystem('outputs_2_screen',  Outputs_2_Screen(), promotes=['*'])
        self.add_subsystem('conv_plots',        Convergence_Trends_Opt(folder_output = folder_output, optimization_log = 'log_opt_' + RefBlade['config']['name']))
        

def set_web3_offset(blade):
    # User Routine to set the 3rd web offset to be at 90% chord for the NREL 15MW
    web_name     = 'third_web'
    web_position = 0.9
    # ------------------------

    web_idx  = [idx for idx, web in enumerate(blade['st']['webs']) if web['name']==web_name][0]

    r_in          = blade['ctrl_pts']['r_in']
    chord_in      = blade['ctrl_pts']['chord_in']
    p_le_spline   = PchipInterpolator(blade['pf']['s'], blade['pf']['p_le'])
    p_le_in       = p_le_spline(r_in)
    
    offset_in     = [chord_i*(web_position-p_le_i) for p_le_i, chord_i in zip(p_le_in, chord_in)]
    offset_spline = PchipInterpolator(r_in, offset_in)
    offset        = offset_spline(blade['pf']['s'])

    blade['st']['webs'][web_idx]['offset_x_pa']['grid']   = blade['pf']['s']
    blade['st']['webs'][web_idx]['offset_x_pa']['values'] = offset

    return blade


def initialize_problem(Analysis_Level, optFlag=False):
    
    # Initialize blade design
    refBlade = ReferenceBlade()
    refBlade.verbose  = True
    refBlade.NINPUT       = 8
    Nsection_Tow          = 19
    refBlade.NPTS         = 30
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
    #prob.model=Optimize_MonopileTurbine(RefBlade=blade, Nsection_Tow=Nsection_Tow, folder_output=folder_output, user_update_routine=set_web3_offset, FASTpref=FASTpref)
    prob.model=Optimize_MonopileTurbine(RefBlade=blade, Nsection_Tow=Nsection_Tow, folder_output=folder_output, FASTpref=FASTpref)
    prob.model.nonlinear_solver = om.NonlinearRunOnce()
    prob.model.linear_solver    = om.DirectSolver()

    if optFlag:
        # --- Driver ---
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'CONMIN'
        # prob.driver.opt_settings['ITMAX']     = 2
        # ----------------------

        # --- Objective ---
        prob.model.add_objective('lcoe')
        # ----------------------

        # --- Design Variables ---
        indices_no_root         = range(2,refBlade.NINPUT)
        indices_no_root_no_tip  = range(2,refBlade.NINPUT-1)
        prob.model.add_design_var('chord_in',    indices = indices_no_root_no_tip, lower=0.5,      upper=7.0)
        prob.model.add_design_var('theta_in',    indices = indices_no_root,        lower=-5.0,     upper=20.0)
        prob.model.add_design_var('sparT_in',    indices = indices_no_root_no_tip, lower=0.001,    upper=0.200)
        prob.model.add_design_var('control_tsr',                                   lower=6.000,    upper=11.00)
        # prob.model.add_design_var('tower_section_height', lower=5.0,  upper=80.0)
        # prob.model.add_design_var('tower_outer_diameter', lower=3.87, upper=10.0)
        # prob.model.add_design_var('tower_wall_thickness', lower=4e-3, upper=2e-1)
        # ----------------------
        
        # --- Constraints ---
        # Rotor
        prob.model.add_constraint('tip_deflection_ratio',     upper=1.0)  
        #prob.model.add_constraint('AEP',                      lower=prob_ref['AEP'])
        # Tower
        # prob.model.add_constraint('tow.height_constraint',    lower=-1e-2,upper=1.e-2)
        # prob.model.add_constraint('tow.post.stress',          upper=1.0)
        # prob.model.add_constraint('tow.post.global_buckling', upper=1.0)
        # prob.model.add_constraint('tow.post.shell_buckling',  upper=1.0)
        # prob.model.add_constraint('tow.weldability',          upper=0.0)
        # prob.model.add_constraint('tow.manufacturability',    lower=0.0)
        # prob.model.add_constraint('frequencyNP_margin',       upper=0.)
        # prob.model.add_constraint('frequency1P_margin',       upper=0.)
        # prob.model.add_constraint('ground_clearance',         lower=20.0)
        # ----------------------
        
        # --- Recorder ---
        filename_opt_log = folder_output + os.sep + 'log_opt_' + blade['config']['name']
        
        prob.driver.add_recorder(om.SqliteRecorder(filename_opt_log))
        prob.driver.recording_options['includes'] = ['AEP','total_blade_cost','lcoe','tip_deflection_ratio']
        prob.driver.recording_options['record_objectives']  = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars']     = True
        # ----------------------

    # Initialize variable inputs
    prob.setup()
    prob = Init_MonopileTurbine(prob, blade, Nsection_Tow = Nsection_Tow, Analysis_Level = Analysis_Level, fst_vt = fst_vt)

    prob['tilt']                    = 6.
    prob['overhang']                = 10.454
    prob['hub_cm']                  = np.array([-10.685, 0.0, 5.471])
    prob['nac_cm']                  = np.array([-5.718, 0.0, 4.048])
    prob['hub_I']                   = np.array([1382171.187, 2169261.099, 2160636.794, 0.0, 0.0, 0.0])
    prob['nac_I']                   = np.array([13442265.552, 21116729.439, 18382414.385, 0.0, 0.0, 0.0])
    prob['hub_mass']                = 140e3
    prob['nac_mass']                = 797.275e3
    prob['hss_mass']                = 0.0
    prob['lss_mass']                = 19.504e3
    prob['cover_mass']              = 0.0
    prob['pitch_system_mass']       = 50e3
    prob['platforms_mass']          = 0.0
    prob['spinner_mass']            = 0.0
    prob['transformer_mass']        = 0.0
    prob['vs_electronics_mass']     = 0.0
    prob['yaw_mass']                = 100e3
    prob['gearbox_mass']            = 0.0
    prob['generator_mass']          = 226.7e3+145.25e3
    prob['bedplate_mass']           = 39.434e3
    prob['main_bearing_mass']       = 4.699e3
    prob['significant_wave_height'] = 4.52
    prob['significant_wave_period'] = 9.45
    prob['monopile']                = True
    prob['foundation_height']       = -30.
    prob['water_depth']             = 30.
    prob['suctionpile_depth']       = 45.
    prob['wind_reference_height']   = 150.
    prob['hub_height']              = 150.
    prob['tower_section_height']    = np.array([5., 5., 5., 5., 5., 5., 5., 5., 5., 13.,  13.,  13.,  13.,  13.,  13.,  13.,  13.,  13., 12.58244309])
    prob['tower_outer_diameter']    = np.array([10., 9.86825264, 9.86911476, 9.86825264, 9.86911476, 9.86825264, 9.86911476, 9.86825264, 9.4022364, 9., 9., 9., 9., 9., 9., 9., 9., 9., 7.28602076, 6.5])
    prob['tower_wall_thickness']    = np.array([0.05663942, 0.05532156, 0.05340554, 0.05146051, 0.04948683, 0.05356181, 0.05003754, 0.04678877, 0.04671201, 0.075, 0.06938931, 0.06102308, 0.05249128, 0.0438327, 0.03514112, 0.02844383, 0.02447982, 0.02349068, 0.02922751])
    prob['tower_buckling_length']   = 15.0
    prob['transition_piece_mass']   = 100e3
    prob['transition_piece_height'] = 15.0

    return prob, blade


def run_problem(prob, optFlag=False):

    # Run initial condition no matter what
    print('Running at Initial Position:')
    prob.run_model()

    # Screen outputs
    print('rna_mass', prob['tow.pre.mass'])
    print('rna_I', prob['tow.pre.mI'])
    print('rna_F', prob['tow.pre.rna_F'])
    print('rna_M', prob['tow.pre.rna_M'])
    print('rna_cg', prob['rna_cg'])
    print('Uref', prob['tow.wind.Uref'])
    print('frequencies', prob['tow.post.structural_frequencies'])
    #print('stress', prob['tow.post.stress'])
    #print('local buckling', prob['tow.post.shell_buckling'])
    #print('shell buckling', prob['tow.post.global_buckling'])

    print('AEP =',                      prob['AEP'])
    print('diameter =',                 prob['diameter'])
    print('ratedConditions.V =',        prob['rated_V'])
    print('ratedConditions.Omega =',    prob['rated_Omega'])
    print('ratedConditions.pitch =',    prob['rated_pitch'])
    print('ratedConditions.T =',        prob['rated_T'])
    print('ratedConditions.Q =',        prob['rated_Q'])
    print('mass_one_blade =',           prob['mass_one_blade'])
    print('mass_all_blades =',          prob['mass_all_blades'])
    print('I_all_blades =',             prob['I_all_blades'])
    print('freq =',                     prob['freq_pbeam'])
    print('tip_deflection =',           prob['tip_deflection'])
    print('root_bending_moment =',      prob['root_bending_moment'])
    print('moments at the hub =',       prob['Mxyz_total'])
    print('blade cost =',               prob['total_blade_cost'])

    # Complete data dump
    #prob.model.list_inputs(units=True)
    #prob.model.list_outputs(units=True)
    
    
    if optFlag:
        prob_ref = copy.deepcopy(prob)
        print('Running Optimization:')
        print('N design var: ', 2*len(indices_no_root_no_tip) + len(indices_no_root) + 1)
        prob.model.approx_totals()
        prob.run_driver()

        # --- Save output .yaml ---
        refBlade.write_ontology(fname_output, prob['blade_out'], refBlade.wt_ref)
        shutil.copyfile(fname_input,  folder_output + os.sep + fname_output)

        # ----------------------
        # --- Outputs plotting ---
        print('AEP:         \t\t\t %f\t%f GWh \t Difference: %f %%' % (prob_ref['AEP']*1e-6, prob['AEP']*1e-6, (prob['AEP']-prob_ref['AEP'])/prob_ref['AEP']*100.))
        print('LCoE:        \t\t\t %f\t%f USD/MWh \t Difference: %f %%' % (prob_ref['lcoe']*1.e003, prob['lcoe']*1.e003, (prob['lcoe']-prob_ref['lcoe'])/prob_ref['lcoe']*100.))
        print('Blade cost:  \t\t\t %f\t%f USD \t Difference: %f %%' % (prob_ref['total_blade_cost'], prob['total_blade_cost'], (prob['total_blade_cost']-prob_ref['total_blade_cost'])/prob_ref['total_blade_cost']*100.))
        print('Blade mass:  \t\t\t %f\t%f kg  \t Difference: %f %%' % (prob_ref['total_blade_mass'], prob['total_blade_mass'], (prob['total_blade_mass']-prob_ref['total_blade_mass'])/prob_ref['total_blade_mass']*100.))
        print('Tower cost:  \t\t\t %f\t%f USD \t Difference: %f %%' % (prob_ref['tower_cost'], prob['tower_cost'], (prob['tower_cost']-prob_ref['tower_cost'])/prob_ref['tower_cost']*100.))
        print('Tower mass:  \t\t\t %f\t%f kg  \t Difference: %f %%' % (prob_ref['tower_mass'], prob['tower_mass'], (prob['tower_mass']-prob_ref['tower_mass'])/prob_ref['tower_mass']*100.))
        # ----------------------
    
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
    towDF['Flapwise inertia [kg.m]'] = towDF['Mass Density [kg/m]'] * I/A
    towDF['Edgewise inertia [kg.m]'] = towDF['Mass Density [kg/m]'] * I/A
    towDF['Flapwise stiffness [N.m^2]'] = 2e11 * I
    towDF['Edgewise stiffness [N.m^2]'] = 2e11 * I
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
    ax.plot(towDF['Height [m]'], towDF['Flapwise inertia [kg.m]'], linewidth=2)
    plt.xlabel('Location [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Flap/Edge inertia [kg.m]', fontsize=14, fontweight='bold')
    fig_name = 'tower_flapedge-inertia'
    format_save(fig, fig_name)

    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(towDF['Height [m]'], towDF['Flapwise stiffness [N.m^2]'], linewidth=2)
    ax.plot(towDF['Height [m]'], towDF['Torsional stiffness [N.m^2]'], linewidth=2)
    ax.legend(('Flap/edge','Torsional'), loc='best')
    plt.xlabel('Location [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Stiffness [N.m^2]', fontsize=14, fontweight='bold')
    fig_name = 'tower_sitffness'
    format_save(fig, fig_name)

    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(towDF['Height [m]'], towDF['Axial stiffness [N]'], linewidth=2)
    plt.xlabel('Location [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Axial Stiffness [N]', fontsize=14, fontweight='bold')
    fig_name = 'tower_axial_sitffness'
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
    fig_name = 'blade_sitffness'
    format_save(fig, fig_name)

    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(xx, prob['EA'], linewidth=2)
    plt.xlabel('Nondimensional Blade Span (r/R)', fontsize=14, fontweight='bold')
    plt.ylabel('Axial Stiffness [N]', fontsize=14, fontweight='bold')
    fig_name = 'blade_axial_sitffness'
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
    
    # Tower plot
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
    
        
    # Write tabular data to xlsx
    myobj = RWT_Tabular(fname_input, towDF=towDF, rotDF=perfDF)
    myobj.write_all()


if __name__ == "__main__":
    # Set optimization
    optFlag = False

    # Initialize output container
    if not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    # Set FAST integration
    # 0: Run CCBlade; 1: Update FAST model at each iteration but do not run; 2: Run FAST w/ ElastoDyn; 3: (Not implemented) Run FAST w/ BeamDyn
    Analysis_Level = 0

    # Seed inputs
    prob, blade = initialize_problem(Analysis_Level, optFlag=optFlag)

    # Run the analysis
    prob = run_problem(prob, optFlag=False)    

    # Generate output plots, tables, and Excel sheet
    postprocess(prob, blade)
