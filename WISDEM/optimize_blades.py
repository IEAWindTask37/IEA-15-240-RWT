from __future__ import print_function
import numpy as np
from scipy.interpolate import PchipInterpolator
import os, shutil, copy, time, sys
import matplotlib.pyplot as plt
from pprint import pprint
from openmdao.api import IndepVarComp, ExplicitComponent, Group, Problem, ScipyOptimizeDriver, SqliteRecorder, NonlinearRunOnce, DirectSolver, CaseReader, ExecComp
try:
    from openmdao.api import pyOptSparseDriver
except:
    pass
from wisdem.rotorse.rotor import RotorSE, Init_RotorSE_wRefBlade
from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade
# from wisdem.rotorse.rotor_fast import eval_unsteady
from wisdem.rotorse.rotor_visualization import plot_lofted
from wisdem.towerse.tower import TowerSE
from wisdem.commonse import NFREQ
# from wisdem.commonse.rna import RNA
from wisdem.commonse.environment import PowerWind, LogWind
from wisdem.commonse.turbine_constraints import TurbineConstraints
from wisdem.turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from wisdem.plant_financese.plant_finance import PlantFinance
from wisdem.drivetrainse.drivese_omdao import DriveSE
from wisdem.assemblies.land_based.land_based_noGenerator_noBOS_lcoe import LandBasedTurbine, Init_LandBasedAssembly
from wisdem.assemblies.fixed_bottom.monopile_assembly_turbine import MonopileTurbine, Init_MonopileTurbine
from wisdem.commonse.mpi_tools import MPI

from wisdem.aeroelasticse.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7




# Class to print outputs on screen
class Outputs_2_Screen(ExplicitComponent):
    def setup(self):
        
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
        self.add_input('freq_pbeam',           val=np.zeros(NFREQ), units = 'Hz')
        self.add_input('freq_distance',        val=0.0)

        
    def compute(self, inputs, outputs):
        print('########################################')
        print('Optimization variables')
        print('TSR:         {:8.3f} -'.format(inputs['control_tsr'][0]))
        print('')
        print('Constraints')
        print('Max TD:      {:8.3f} m'.format(inputs['tip_deflection'][0]))
        print('TD ratio:    {:8.10f} -'.format(inputs['tip_deflection_ratio'][0]))
        print('Blade Freq:', inputs['freq_pbeam'])
        print('Freq Ratio:  {:8.5f}'.format(inputs['freq_distance'][0]))
        print('')
        print('Objectives')
        print('AEP:         {:8.10f} GWh'.format(inputs['AEP'][0]))
        print('Blade mass:  {:8.3f} kg'.format(inputs['mass_one_blade'][0]))
        print('Blade cost:  {:8.3f} $'.format(inputs['total_blade_cost'][0]))
        print('Tower mass:  {:8.3f} kg'.format(inputs['tower_mass'][0]))
        print('Tower cost:  {:8.3f} $'.format(inputs['tower_cost'][0]))
        print('LCoE:        {:8.3f} $/MWh'.format(inputs['lcoe'][0]))
        print('########################################')


class blade_freq_check(ExplicitComponent):
    def setup(self):
        self.add_input('freq_pbeam', val=np.ones(5), units = 'Hz')
        self.add_output('freq_check_out', val=1.)
        
    def compute(self, inputs, outputs):
        print(inputs['freq_pbeam'])
        outputs['freq_check_out'] = inputs['freq_pbeam'][1]/inputs['freq_pbeam'][0]

class Convergence_Trends_Opt(ExplicitComponent):
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
                fig_name = 'Convergence_trend_' + param + '.png'
                fig.savefig(folder_output + fig_name)
                plt.close(fig)


# Group to link the openmdao components
class Optimize_MonopileTurbine(Group):

    def initialize(self):
        self.options.declare('RefBlade')
        self.options.declare('folder_output' ,  default='')
        self.options.declare('FASTpref',        default={})
        self.options.declare('Nsection_Tow',    default = 6)
        self.options.declare('VerbosityCosts',  default = False)
        self.options.declare('user_update_routine',     default=None)
        
    def setup(self):
        RefBlade             = self.options['RefBlade']
        folder_output        = self.options['folder_output']
        Nsection_Tow         = self.options['Nsection_Tow']
        VerbosityCosts       = self.options['VerbosityCosts']
        user_update_routine  = self.options['user_update_routine']
        FASTpref             = self.options['FASTpref']
    
        self.add_subsystem('lb_wt', MonopileTurbine(RefBlade=blade, Nsection_Tow = Nsection_Tow, VerbosityCosts = VerbosityCosts, FASTpref=FASTpref), promotes=['*']) #, user_update_routine=user_update_routine
        # Post-processing
        self.add_subsystem('outputs_2_screen',  Outputs_2_Screen(), promotes=['*'])
        self.add_subsystem('conv_plots',        Convergence_Trends_Opt(folder_output = folder_output, optimization_log = 'log_opt_' + RefBlade['config']['name']))
        


if __name__ == "__main__":
    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
    
    optFlag       = True

    WT_input      = "IEA-15-240-RWT.yaml"
    WT_output     = "IEA-15-240-RWT_out.yaml"

    folder_input  = "/mnt/c/Users/egaertne/IEA-15-240-RWT/WISDEM/"
    folder_output = '/mnt/c/Users/egaertne/IEA-15-240-RWT/WISDEM/results/IEA15MW_out/'
    schema        = 'IEAontology_schema.yaml'
    
    fname_schema  = folder_input + schema
    fname_input   = folder_input + WT_input
    fname_output  = folder_output + WT_output
    
    if not os.path.isdir(folder_output) and rank==0:
        os.mkdir(folder_output)

    Analysis_Level        = 0 # 0: Run CCBlade; 1: Update FAST model at each iteration but do not run; 2: Run FAST w/ ElastoDyn; 3: (Not implemented) Run FAST w/ BeamDyn
    # Initialize blade design
    refBlade = ReferenceBlade()
    if rank == 0:
        refBlade.verbose  = True
    else:
        refBlade.verbose  = False
    refBlade.NINPUT       = 8
    Nsection_Tow          = 12
    refBlade.NPTS         = 30
    refBlade.spar_var     = ['Spar_cap_ss', 'Spar_cap_ps'] # SS, then PS
    refBlade.te_var       = 'TE_reinforcement'
    refBlade.le_var       = 'le_reinf'
    refBlade.validate     = False
    refBlade.fname_schema = fname_schema
    blade = refBlade.initialize(fname_input)
    
    FASTpref                        = {}
    FASTpref['Analysis_Level']      = Analysis_Level
    # Set FAST Inputs
    # if Analysis_Level >= 1:
    #     # File management
    #     FASTpref['FAST_ver']            = 'OpenFAST'
    #     FASTpref['dev_branch']          = True
    #     FASTpref['FAST_exe']            = 'openfast'
    #     FASTpref['FAST_directory']      = '/mnt/c/Users/egaertne/IEA-15-240-RWT/OpenFAST/'
    #     FASTpref['FAST_InputFile']      = 'IEA-15-240-RWT.fst'
    #     FASTpref['Turbsim_exe']         = "turbsim"
    #     FASTpref['FAST_namingOut']      = 'IEA-15-240-RWT'
    #     FASTpref['FAST_runDirectory']   = 'temp/' + FASTpref['FAST_namingOut']
        
    #     # Run Settings
    #     FASTpref['cores']               = 1
    #     FASTpref['debug_level']         = 2 # verbosity: set to 0 for quiet, 1 & 2 for increasing levels of output

    #     # DLCs
    #     FASTpref['DLC_gust']            = None      # Max deflection
    #     FASTpref['DLC_extrm']           = None      # Max strain
    #     FASTpref['DLC_turbulent']       = None
    #     FASTpref['DLC_powercurve']      = None      # AEP

    #     # Initialize, read initial FAST files to avoid doing it iteratively
    #     fast = InputReader_OpenFAST(FAST_ver=FASTpref['FAST_ver'], dev_branch=FASTpref['dev_branch'])
    #     fast.FAST_InputFile = FASTpref['FAST_InputFile']
    #     fast.FAST_directory = FASTpref['FAST_directory']
    #     fast.execute()
    #     fst_vt = fast.fst_vt
    # else:
    fst_vt = {}
    
    # Initialize and execute OpenMDAO problem with input data
    if MPI:
        num_par_fd = MPI.COMM_WORLD.Get_size()
        prob_ref   = Problem(model=Group(num_par_fd=num_par_fd))
        prob_ref.model.approx_totals(method='fd')
        prob_ref.model.add_subsystem('comp', Optimize_MonopileTurbine(RefBlade=blade, Nsection_Tow = Nsection_Tow, VerbosityCosts = False, folder_output = folder_output), promotes=['*'])
    else:
        prob_ref = Problem()
        prob_ref.model=Optimize_MonopileTurbine(RefBlade=blade, Nsection_Tow = Nsection_Tow, VerbosityCosts = False, folder_output = folder_output, FASTpref = FASTpref)

    prob_ref.setup()
    prob_ref = Init_MonopileTurbine(prob_ref, blade, Nsection_Tow = Nsection_Tow, Analysis_Level = Analysis_Level, fst_vt = fst_vt)
    prob_ref['gust_stddev'] = 3
    prob_ref['tower_section_height']           = (prob_ref['hub_height'] - prob_ref['foundation_height']) / Nsection_Tow * np.ones(Nsection_Tow)
    prob_ref['drive.shaft_angle']              = np.radians(6.)
    prob_ref['overhang']                       = 11.014
    prob_ref['drive.distance_hub2mb']          = 3.819

    prob_ref['foundation_height']              = -30.
    prob_ref['water_depth']                    = 30.
    prob_ref['wind_reference_height']          = 150.
    prob_ref['hub_height']                     = 150.
    prob_ref['tower_outer_diameter']           = np.array([10., 10., 10., 10., 9.692655, 9.312475, 8.911586, 8.532367, 8.082239, 7.621554, 7.286144, 6.727136, 6.3])
    prob_ref['tower_section_height']           = (prob_ref['hub_height'] - prob_ref['foundation_height']) / Nsection_Tow * np.ones(Nsection_Tow)
    prob_ref['tower_wall_thickness']           = np.array([0.04224176, 0.04105759, 0.0394965, 0.03645589, 0.03377851, 0.03219233, 0.03070819, 0.02910109, 0.02721289, 0.02400931, 0.0208264, 0.02399756])
    prob_ref['nostallconstraint.min_s']        = 0.25  # The stall constraint is only computed from this value (nondimensional coordinate along blade span) to blade tip
    prob_ref['nostallconstraint.stall_margin'] = 3.0   # Values in deg of stall margin
    # prob_ref['dynamic_amplification']          = 1.075 # steady/dynamic tuning factor
    

    prob_ref.model.nonlinear_solver = NonlinearRunOnce()
    prob_ref.model.linear_solver    = DirectSolver()
    if rank == 0:
        print('Running at Initial Position:')
    prob_ref.run_driver()
    
    # refBlade.write_ontology(fname_output, prob_ref['blade_out'], refBlade.wt_ref)
    
    print('AEP =',                      prob_ref['AEP'])
    print('diameter =',                 prob_ref['diameter'])
    print('ratedConditions.V =',        prob_ref['rated_V'])
    print('ratedConditions.Omega =',    prob_ref['rated_Omega'])
    print('ratedConditions.pitch =',    prob_ref['rated_pitch'])
    print('ratedConditions.T =',        prob_ref['rated_T'])
    print('ratedConditions.Q =',        prob_ref['rated_Q'])
    print('mass_one_blade =',           prob_ref['mass_one_blade'])
    print('mass_all_blades =',          prob_ref['mass_all_blades'])
    print('I_all_blades =',             prob_ref['I_all_blades'])
    print('freq =',                     prob_ref['freq_pbeam'])
    print('tip_deflection =',           prob_ref['tip_deflection'])
    print('root_bending_moment =',      prob_ref['root_bending_moment'])
    print('moments at the hub =',       prob_ref['Mxyz_total'])
    print('blade cost =',               prob_ref['total_blade_cost'])
    print('Cp_aero =',                  max(prob_ref['Cp_aero']))

    
    # Angle of attack and stall angle
    faoa, axaoa = plt.subplots(1,1,figsize=(5.3, 4))
    axaoa.plot(prob_ref['r'], prob_ref['nostallconstraint.aoa_along_span'], label='Initial aoa')
    axaoa.plot(prob_ref['r'], prob_ref['nostallconstraint.stall_angle_along_span'], '.', label='Initial stall')
    axaoa.legend(fontsize=12)
    plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Angle [deg]', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(color=[0.8,0.8,0.8], linestyle='--')
    plt.subplots_adjust(bottom = 0.15, left = 0.15)
    fig_name = 'aoa.png'
    faoa.savefig(folder_output + fig_name)

    # plt.show()    
    
    # Run an optimization
    if optFlag:
        if MPI:
            num_par_fd = MPI.COMM_WORLD.Get_size()
            prob = Problem(model=Group(num_par_fd=num_par_fd))
            prob.model.approx_totals(method='fd')
            prob.model.add_subsystem('comp', Optimize_MonopileTurbine(RefBlade=blade, Nsection_Tow = Nsection_Tow, folder_output = folder_output), promotes=['*'])
        else:
            prob = Problem()
            prob.model=Optimize_MonopileTurbine(RefBlade=blade, Nsection_Tow = Nsection_Tow, folder_output = folder_output)
        
        # --- Driver ---
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'CONMIN'
        prob.driver.opt_settings['ITMAX']     = 15
        prob.driver.opt_settings['IPRINT']     = 4
        # ----------------------

        # --- Objective ---
        # prob.model.add_objective('AEP', scaler = -1.)
        prob.model.add_objective('mass_one_blade')
        # ----------------------

        # --- Design Variables ---
        indices_no_root         = range(2,refBlade.NINPUT)
        indices_no_root_no_tip  = range(2,refBlade.NINPUT-1)
        indices_no_max_chord    = range(3,refBlade.NINPUT)
        prob.model.add_design_var('sparT_in',    indices = indices_no_root_no_tip, lower=0.001,    upper=0.200)
        # prob.model.add_design_var('chord_in',    indices = indices_no_max_chord,   lower=0.5,      upper=7.0)
        # prob.model.add_design_var('theta_in',    indices = indices_no_root,        lower=-7.5,     upper=20.0)
        # prob.model.add_design_var('teT_in', lower=prob_ref['teT_in']*0.5, upper=0.1)
        # prob.model.add_design_var('leT_in', lower=prob_ref['leT_in']*0.5, upper=0.1)
        # prob.model.add_design_var('control_tsr',                                   lower=6.000,    upper=11.00)
        # prob.model.add_design_var('tower_section_height', lower=5.0,  upper=80.0)
        # prob.model.add_design_var('tower_outer_diameter', lower=3.87, upper=10.0)
        # prob.model.add_design_var('tower_wall_thickness', lower=4e-3, upper=2e-1)
        # ----------------------
        
        # --- Constraints ---
        # Rotor
        prob.model.add_constraint('tip_deflection_ratio',     upper=1.0)  
        # prob.model.add_constraint('no_stall_constraint',      upper=1.0)  
        # prob.model.add_constraint('mass_one_blade',           upper=prob_ref['mass_one_blade']*1.02)  
        # prob.model.add_constraint('freq_check', lower=1.1)
        # prob.model.add_constraint('AEP',                      lower=prob_ref['AEP'])

        # prob.model.add_subsystem('freq_check', blade_freq_check(), promotes=['freq_check_out'])
        # prob.model.connect('freq_pbeam', 'freq_check.freq_pbeam')#, src_indices=[0])
        # prob.model.add_subsystem('freq_check', ExecComp('f_check=freq_pbeam1/freq_pbeam0', freq_pbeam0={'units':'Hz'}, freq_pbeam1={'units':'Hz'}))
        # prob.model.connect('freq_pbeam', 'freq_check.freq_pbeam0', src_indices=[0])
        # prob.model.connect('freq_pbeam', 'freq_check.freq_pbeam1', src_indices=[1])
        # prob.model.add_constraint('tip_deflection', upper=prob_ref['tip_deflection'])

        # Tower
        # prob.model.add_constraint('tow.height_constraint',    lower=-1e-2,upper=1.e-2)
        # prob.model.add_constraint('tow.post.stress',          upper=1.0)
        # prob.model.add_constraint('tow.post.global_buckling', upper=1.0)
        # prob.model.add_constraint('tow.post.shell_buckling',  upper=1.0)
        # prob.model.add_constraint('tow.weldability',          upper=0.0)
        # prob.model.add_constraint('tow.manufacturability',    lower=0.0)
        # prob.model.add_constraint('frequencyNP_margin',       upper=0.)
        # prob.model.add_constraint('frequency1P_margin',       upper=0.)
        # ----------------------
        
        # --- Recorder ---
        filename_opt_log = folder_output + 'log_opt_' + blade['config']['name']
        
        prob.driver.add_recorder(SqliteRecorder(filename_opt_log))
        prob.driver.recording_options['includes'] = ['AEP','total_blade_cost','lcoe','tip_deflection_ratio','theta_in']
        prob.driver.recording_options['record_objectives']  = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars']     = True
        # ----------------------
        
        # --- Run ---
        prob.setup()
        prob = Init_MonopileTurbine(prob, blade, Nsection_Tow = Nsection_Tow)
        prob['gust_stddev'] = 3
        prob['tower_section_height']           = (prob_ref['hub_height'] - prob_ref['foundation_height']) / Nsection_Tow * np.ones(Nsection_Tow)
        prob['drive.shaft_angle']              = np.radians(6.)
        prob['overhang']                       = 11.014
        prob['drive.distance_hub2mb']          = 3.819
        
        prob['foundation_height']              = -30.
        prob['water_depth']                    = 30.
        prob['wind_reference_height']          = 150.
        prob['hub_height']                     = 150.
        prob['tower_outer_diameter']           = np.array([10., 10., 10., 10., 9.692655, 9.312475, 8.911586, 8.532367, 8.082239, 7.621554, 7.286144, 6.727136, 6.3])
        prob['tower_section_height']           = (prob_ref['hub_height'] - prob_ref['foundation_height']) / Nsection_Tow * np.ones(Nsection_Tow)
        prob['tower_wall_thickness']           = np.array([0.04224176, 0.04105759, 0.0394965, 0.03645589, 0.03377851, 0.03219233, 0.03070819, 0.02910109, 0.02721289, 0.02400931, 0.0208264, 0.02399756])
        prob['nostallconstraint.min_s']        = 0.25  # The stall constraint is only computed from this value (nondimensional coordinate along blade span) to blade tip
        prob['nostallconstraint.stall_margin'] = 3.0   # Values in deg of stall margin
    
        prob.model.nonlinear_solver = NonlinearRunOnce()
        prob.model.linear_solver = DirectSolver()
        if rank == 0:
            print('Running Optimization:')
            print('N design var: ', 2*len(indices_no_root_no_tip) + len(indices_no_root) + 1)
        if not MPI:
            prob.model.approx_totals()
        prob.run_driver()
        # ----------------------

        if rank == 0:
            # --- Save output .yaml ---
            refBlade.write_ontology(fname_output, prob['blade_out'], refBlade.wt_ref)
            shutil.copyfile(fname_input,  folder_output + WT_input)
            # ----------------------
            # --- Outputs plotting ---
            print('AEP:         \t\t\t %f\t%f GWh \t Difference: %f %%' % (prob_ref['AEP']*1e-6, prob['AEP']*1e-6, (prob['AEP']-prob_ref['AEP'])/prob_ref['AEP']*100.))
            print('LCoE:        \t\t\t %f\t%f USD/MWh \t Difference: %f %%' % (prob_ref['lcoe']*1.e003, prob['lcoe']*1.e003, (prob['lcoe']-prob_ref['lcoe'])/prob_ref['lcoe']*100.))
            print('Blade cost:  \t\t\t %f\t%f USD \t Difference: %f %%' % (prob_ref['total_blade_cost'], prob['total_blade_cost'], (prob['total_blade_cost']-prob_ref['total_blade_cost'])/prob_ref['total_blade_cost']*100.))
            print('Blade mass:  \t\t\t %f\t%f kg  \t Difference: %f %%' % (prob_ref['total_blade_mass'], prob['total_blade_mass'], (prob['total_blade_mass']-prob_ref['total_blade_mass'])/prob_ref['total_blade_mass']*100.))
            print('Tower cost:  \t\t\t %f\t%f USD \t Difference: %f %%' % (prob_ref['tower_cost'], prob['tower_cost'], (prob['tower_cost']-prob_ref['tower_cost'])/prob_ref['tower_cost']*100.))
            print('Tower mass:  \t\t\t %f\t%f kg  \t Difference: %f %%' % (prob_ref['tower_mass'], prob['tower_mass'], (prob['tower_mass']-prob_ref['tower_mass'])/prob_ref['tower_mass']*100.))
            # ----------------------
            
            # Theta
            ft, axt = plt.subplots(1,1,figsize=(5.3, 4))
            axt.plot(prob_ref['r'], prob_ref['theta'], label='Initial')
            axt.plot(prob_ref['r_in'], prob_ref['theta_in'], '.')
            axt.plot(prob['r'], prob['theta'], label='Optimized')
            axt.plot(prob['r_in'], prob['theta_in'], '.')
            axt.legend(fontsize=12)
            plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
            plt.ylabel('Twist [deg]', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            plt.subplots_adjust(bottom = 0.15, left = 0.15)
            fig_name = 'theta.png'
            ft.savefig(folder_output + fig_name)
            
            # Angle of attack and stall angle
            faoa, axaoa = plt.subplots(1,1,figsize=(5.3, 4))
            axaoa.plot(prob_ref['r'], prob_ref['nostallconstraint.aoa_along_span'], label='Initial aoa')
            axaoa.plot(prob_ref['r'], prob_ref['nostallconstraint.stall_angle_along_span'], '.', label='Initial stall')
            axaoa.plot(prob['r'], prob['nostallconstraint.aoa_along_span'], label='Optimized aoa')
            axaoa.plot(prob['r'], prob['nostallconstraint.stall_angle_along_span'], '.', label='Optimized stall')
            axaoa.legend(fontsize=12)
            plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
            plt.ylabel('Angle [deg]', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            plt.subplots_adjust(bottom = 0.15, left = 0.15)
            fig_name = 'aoa.png'
            faoa.savefig(folder_output + fig_name)

            plt.show()
