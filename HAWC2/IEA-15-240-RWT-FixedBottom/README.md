# HAWC2 models

## HTC file(s)

Only one htc file is provided in the repo: a 100s, steady-wind
simulation with constant shear using the flexible FPM model.
See "Blade Models" below for more information on the options
for blade modeling.

For convenience, we have provided helper Python scripts in `make_htc`
to create three types of htc files from the base file: (1) a HAWC2 step-wind
from cut-in to cut-out, (2) a HAWC2 turbulence simulation, and (3) a 
HAWCStab2/HAWC2S input file. **Note** that if you want to run with the
no-torsion model, you should change the blade model and controller parameters
in the base file (see "Using no-torsion model" below) before running the
scripts. The Python scripts require Python >= 3.6 and the Wind Energy
Toolbox, which can be installed with `pip install wetb`.


## Blade models

There are three blade structural files included in this repo:  
 1. **Fully populated matrix**: `IEA_15MW_RWT_Blade_st_fpm.dat`.
    This is HAWC2's Timoshenko model with a 6x6 stiffness matrix. 
    There are three sets in the file: flexible, no-torsion, and stiff.

 2. **Classical Timoshenko**: `IEA_15MW_RWT_Blade_st_nofpm.dat`.
    Same properties as FPM but modelled using HAWC2's classical Timoshenko beam.
    There are three sets in the file: flexible, no-torsion, and stiff.

 3. **BTC blade**: `IEA_15MW_RWT_BTC8deg_Blade_st.dat`.
    A bend-twist couple blade, designed as part of the Corewind project.
    Only a flexible blade is provided.


The blade models can be changed by toggling the `filename` and `FPM`
options in the blade's `timoshenko_input` block. More instructions
on using the no-torsion model are provided below. Note that neither
controller parameters nor operational data are provided for the BTC
blade, but they can be calculated using HAWCStab2.

To compare HAWC2 with ElastoDyn, you should use the no-torsion
blade model. To compare HAWC2 with BeamDyn, you should use the fully
flexible blade model. The stiff model is provided purely for convenience.

## Using no-torsion model

If using the no-torsion model, in addition to changing the blade
model, you must also update (1) the controller parameters for the HAWC2
simulation and (2) the operational data file for the HAWCStab2 block.
The controller parameters for the no-torsion model are provided in
`control/controller_block_notorsion.txt`. The operational data file
for the no-torsion model is given in `data/operation_nofpm_notorsion.dat`
