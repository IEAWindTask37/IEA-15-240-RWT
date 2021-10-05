# Controller Tuning Script for NREL-5MW Wind Turbine
#  -- Made to run the tools distributed as a part of the ROSCO_Toolbox
import os

#-------------------------------- LOAD INPUT PARAMETERS ---------------------------------#
# Change this for your turbine
this_dir            = os.path.dirname(os.path.abspath(__file__))
parameter_filename  = os.path.join(this_dir,'IEA15MW_detuned.yaml')                         # Name of .yaml input file for the specific turbine




#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------- NOTHING SHOULD NEED TO CHANGE AFTER THIS -----------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#------------------------------------- INITIALIZATION ----------------------------------#
# Import python modules
import matplotlib.pyplot as plt 
import yaml 
# Import ROSCO_toolbox modules 
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox.utilities import write_rotor_performance, write_DISCON

# Initialize parameter dictionaries
turbine_params = {}
control_params = {}

# Load input file contents, put them in some dictionaries to keep things cleaner
inps = yaml.safe_load(open(parameter_filename))
path_params = inps['path_params']
turbine_params = inps['turbine_params']
controller_params = inps['controller_params']

#---------------------------------- DO THE FUN STUFF ------------------------------------#
# Initialiize turbine and controller
turbine         = ROSCO_turbine.Turbine(turbine_params)

# Load Turbine, write rotor performance file if it doesn't exist
if os.path.exists(os.path.join(this_dir,path_params['rotor_performance_filename'])):
    turbine.load_from_fast(path_params['FAST_InputFile'], \
        os.path.join(this_dir,path_params['FAST_directory']), \
            dev_branch=True,rot_source='txt',txt_filename=path_params['rotor_performance_filename'])
else:
    turbine.load_from_fast(path_params['FAST_InputFile'], \
        os.path.join(this_dir,path_params['FAST_directory']), \
            dev_branch=True,rot_source=None, txt_filename=path_params['rotor_performance_filename'])

    write_rotor_performance(turbine,txt_filename=os.path.join(this_dir,path_params['rotor_performance_filename']))
    
# Flap tuning if necessary
if controller_params['Flp_Mode']:
    turbine.load_blade_info()

# Instantiate controller tuning and tune controller
controller      = ROSCO_controller.Controller(controller_params)
controller.tune_controller(turbine)

# Write parameter input file
param_file = 'DISCON_detuned.IN'   
# write_DISCON(turbine,controller,param_file=param_file, txt_filename=os.path.join(this_dir,path_params['rotor_performance_filename']))
write_DISCON(turbine,controller,param_file=param_file, txt_filename=os.path.join('./control/rosco2hawc2/',path_params['rotor_performance_filename']))

# Plot rotor performance 
turbine.Cp.plot_performance()
plt.show()
