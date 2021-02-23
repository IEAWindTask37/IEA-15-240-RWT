# HAWC2 models

## HTC file(s)

Only one htc file is provided in the repo: a 100s, steady-wind
simulation with constant shear using the flexible FPM model.
See "Blade Models" below for more information on the options
for blade modeling.

for convenience, we have provided helper Python scripts to create four
more types of htc files from the base file: (1) a HAWC2 step-wind
from cut-in to cut-out with the same blade model as the base file,
(2) a HAWC2 turbulence simulation with the same blade model as the
base file, and (3) a HAWCStab2/HAWC2S input file with the same blade
model as the base file. To run the scripts, you must have Python >= 3.6
and the Wind Energy Toolbox, which can be installed with `pip install wetb`.


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
options in the blade's `timoshenko_input` block.

**NOTE**: If using the no-torsion model, you must also update the controller
block using the controller parameters provided in
`control/controller_block_notorsion.txt`. No controller parameters are provided
for the BTC blade, but they can be tuned with HAWCStab2.

To compare HAWC2 with ElastoDyn, you should use the no-torsion
blade model. To compare HAWC2 with BeamDyn, you should use the fully
flexible blade model. The stiff model is provided purely for convenience.
