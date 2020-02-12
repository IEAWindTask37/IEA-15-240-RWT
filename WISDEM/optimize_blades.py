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
from wisdem.assemblies.fixed_bottom.monopile_assembly_turbine_nodrive import MonopileTurbine
from wisdem.commonse.mpi_tools import MPI
from wisdem.commonse import NFREQ

from run_model import initialize_variables

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0

# Global inputs and outputs
ontology_dir  = os.path.dirname( os.path.dirname( os.path.realpath(__file__)) ) + os.sep + 'WT_Ontology'
fname_schema  = ontology_dir + os.sep + 'IEAontology_schema.yaml'
fname_input   = ontology_dir + os.sep + 'IEA-15-240-RWT_FineGrid.yaml'
fname_output  = ontology_dir + os.sep + 'IEA-15-240-RWT_out.yaml'
folder_output = os.getcwd() + os.sep + 'outputs'

if not os.path.isdir(folder_output) and rank==0:
    os.mkdir(folder_output)
    

# Class to print outputs on screen
class Outputs_2_Screen(om.ExplicitComponent):
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
        self.add_input('freq_curvefem',           val=np.zeros(NFREQ), units = 'Hz')
        self.add_input('freq_distance',        val=0.0)

        
    def compute(self, inputs, outputs):
        print('########################################')
        print('Optimization variables')
        print('TSR:         {:8.3f} -'.format(inputs['control_tsr'][0]))
        print('')
        print('Constraints')
        print('Max TD:      {:8.3f} m'.format(inputs['tip_deflection'][0]))
        print('TD ratio:    {:8.10f} -'.format(inputs['tip_deflection_ratio'][0]))
        print('Blade Freq:', inputs['freq_curvefem'])
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


class blade_freq_check(om.ExplicitComponent):
    def setup(self):
        self.add_input('freq_curvefem', val=np.ones(5), units = 'Hz')
        self.add_output('freq_check_out', val=1.)
        
    def compute(self, inputs, outputs):
        print(inputs['freq_curvefem'])
        outputs['freq_check_out'] = inputs['freq_curvefem'][1]/inputs['freq_curvefem'][0]


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
                fig_name = 'Convergence_trend_' + param + '.png'
                fig.savefig(folder_output + fig_name)
                plt.close(fig)



# Group to link the openmdao components
class Optimize_MonopileTurbine(om.Group):

    def initialize(self):
        self.options.declare('RefBlade')
        self.options.declare('folder_output' ,  default='')
        self.options.declare('FASTpref',        default={})
        self.options.declare('Nsection_Tow',    default = 6)
        self.options.declare('VerbosityCosts',  default = False)
        
    def setup(self):
        RefBlade             = self.options['RefBlade']
        folder_output        = self.options['folder_output']
        Nsection_Tow         = self.options['Nsection_Tow']
        VerbosityCosts       = self.options['VerbosityCosts']
        FASTpref             = self.options['FASTpref']
    
        self.add_subsystem('lb_wt', MonopileTurbine(RefBlade=RefBlade, Nsection_Tow = Nsection_Tow, VerbosityCosts = VerbosityCosts, FASTpref=FASTpref), promotes=['*'])
        # Post-processing
        self.add_subsystem('outputs_2_screen',  Outputs_2_Screen(), promotes=['*'])
        self.add_subsystem('conv_plots',        Convergence_Trends_Opt(folder_output = folder_output, optimization_log = 'log_opt_' + RefBlade['config']['name']))
        

        
def run_problem(optFlag=False, prob_ref=None):
    
    # Initialize blade design
    refBlade = ReferenceBlade()
    if rank == 0:
        refBlade.verbose  = True
    else:
        refBlade.verbose  = False
    refBlade.NINPUT       = 8
    Nsection_Tow          = 19
    refBlade.NPTS         = 30
    refBlade.spar_var     = ['Spar_cap_ss', 'Spar_cap_ps'] # SS, then PS
    refBlade.te_var       = 'TE_reinforcement'
    refBlade.le_var       = 'le_reinf'
    refBlade.validate     = False
    refBlade.fname_schema = fname_schema
    blade = refBlade.initialize(fname_input)
    
    Analysis_Level             = 0
    FASTpref                   = {}
    FASTpref['Analysis_Level'] = Analysis_Level
    fst_vt                     = {}

    # Initialize and execute OpenMDAO problem with input data
    if MPI:
        num_par_fd = MPI.COMM_WORLD.Get_size()
        prob       = om.Problem(model=om.Group(num_par_fd=num_par_fd))
        prob.model.approx_totals(method='fd')
        prob.model.add_subsystem('comp', Optimize_MonopileTurbine(RefBlade=blade, Nsection_Tow=Nsection_Tow, folder_output=folder_output), promotes=['*'])
    else:
        prob = om.Problem()
        prob.model = Optimize_MonopileTurbine(RefBlade=blade, Nsection_Tow=Nsection_Tow, folder_output=folder_output)
    
    prob.model.nonlinear_solver = om.NonlinearRunOnce()
    prob.model.linear_solver    = om.DirectSolver()

    if optFlag and not prob_ref is None:
        if MPI:
            num_par_fd = MPI.COMM_WORLD.Get_size()
            prob = om.Problem(model=om.Group(num_par_fd=num_par_fd))
            prob.model.approx_totals(method='fd')
            prob.model.add_subsystem('comp', Optimize_MonopileTurbine(RefBlade=blade, Nsection_Tow=Nsection_Tow, folder_output=folder_output), promotes=['*'])
        else:
            prob = om.Problem()
            prob.model=Optimize_MonopileTurbine(RefBlade=blade, Nsection_Tow=Nsection_Tow, folder_output=folder_output)
        
        # --- Driver ---
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer']   = 'CONMIN'
        prob.driver.opt_settings['ITMAX']  = 15
        prob.driver.opt_settings['IPRINT'] = 4
        # ----------------------

        # --- Objective ---
        # prob.model.add_objective('lcoe')
        prob.model.add_objective('AEP', scaler = -1.)
        #prob.model.add_objective('mass_one_blade')
        # ----------------------

        # --- Design Variables ---
        indices_no_root         = range(2,refBlade.NINPUT)
        indices_no_root_no_tip  = range(2,refBlade.NINPUT-1)
        indices_no_max_chord    = range(3,refBlade.NINPUT)
        prob.model.add_design_var('sparT_in',    indices = indices_no_root_no_tip, lower=0.001,    upper=0.200)
        prob.model.add_design_var('chord_in',    indices = indices_no_max_chord,   lower=0.5,      upper=7.0)
        prob.model.add_design_var('theta_in',    indices = indices_no_root,        lower=-7.5,     upper=20.0)
        prob.model.add_design_var('teT_in', lower=prob_ref['teT_in']*0.5, upper=0.1)
        #prob.model.add_design_var('leT_in', lower=prob_ref['leT_in']*0.5, upper=0.1)
        # ----------------------
        
        # --- Constraints ---
        prob.model.add_subsystem('freq_check', blade_freq_check(), promotes=['freq_check_out'])
        prob.model.connect('freq_curvefem', 'freq_check.freq_curvefem')#, src_indices=[0])

        # Rotor
        prob.model.add_constraint('tip_deflection_ratio',     upper=1.0)  
        # prob.model.add_constraint('no_stall_constraint',      upper=1.0)  
        prob.model.add_constraint('freq_check_out', lower=1.1)
        #prob.model.add_constraint('rated_Q',     lower=21.4e6, upper=21.6e6)  
        # prob.model.add_constraint('mass_one_blade',           upper=prob_ref['mass_one_blade']*1.02)  
        prob.model.add_constraint('AEP',                      lower=0.99*prob_ref['AEP'])
        # ----------------------
        
        # --- Recorder ---
        filename_opt_log = folder_output + 'log_opt_' + blade['config']['name']
        
        prob.driver.add_recorder(om.SqliteRecorder(filename_opt_log))
        prob.driver.recording_options['includes'] = ['AEP','total_blade_cost','lcoe','tip_deflection_ratio','mass_one_blade','theta_in']
        prob.driver.recording_options['record_objectives']  = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars']     = True
        # ----------------------

    # Initialize variable inputs
    prob = initialize_variables(prob, blade, Analysis_Level, fst_vt)

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
    print('########################################')

    # Angle of attack and stall angle
    faoa, axaoa = plt.subplots(1,1,figsize=(5.3, 4))
    axaoa.plot(prob['r'], prob['nostallconstraint.aoa_along_span'], label='Initial aoa')
    axaoa.plot(prob['r'], prob['nostallconstraint.stall_angle_along_span'], '.', label='Initial stall')
    axaoa.legend(fontsize=12)
    plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Angle [deg]', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(color=[0.8,0.8,0.8], linestyle='--')
    plt.subplots_adjust(bottom = 0.15, left = 0.15)
    fig_name = 'aoa.png'
    faoa.savefig(folder_output + fig_name)
    
    # Complete data dump
    #prob.model.list_inputs(units=True)
    #prob.model.list_outputs(units=True)
    
    if optFlag:
        if rank == 0:
            print('Running Optimization:')
            print('N design var: ', 2*len(indices_no_root_no_tip) + len(indices_no_root) + 1)
        if not MPI:
            prob.model.approx_totals()
        prob.run_driver()

        if rank == 0:
            # --- Save output .yaml ---
            refBlade.write_ontology(fname_output, prob['blade_out'], refBlade.wt_ref)

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
            ft.savefig(folder_output + fig_name)

            plt.show()
        
    return prob, blade


if __name__ == '__main__':
    prob_ref, blade = run_problem(False)
    prob, blade = run_problem(True, prob_ref)

