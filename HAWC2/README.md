# HAWC2 models

## Folders

The HAWC2 models are organized into folders:  
 * **IEA-15-240-RWT**: Contains partial htc files that are
   loaded/reused in the different models.  
 * **IEA-15-240-RWT-FixedSubstructure**: The model is fixed
   in translation and rotation at the bottom of the tower.
 * **IEA-15-240-RWT-Monopile**: Model with updated RNA not
   yet implemented in HAWC2.
 * **IEA-15-240-RWT-UMaineSemi**: Model with updated RNA not
   yet implemented in HAWC2.

## HTC file(s)

Only one htc file is provided in the model folder: a 100s,
steady-wind simulation with constant shear, no FPM (equivalent
to a BeamDyn model). See "Blade Models" below for more
information on the options for blade modeling.

For convenience, we have provided helper Python scripts in `make_htc`
to create three types of htc files from the base file: (1) a HAWC2 step-wind
from cut-in to cut-out, (2) a HAWC2 turbulence simulation, and (3) a 
HAWCStab2/HAWC2S input file. The Python scripts require Python 
>= 3.6 and the Wind Energy Toolbox, which can be installed with
`pip install wetb`.


## Blade models

There are two types of blade models: standard Timoschenko beam
(default) and fully-populated matrix (FPM). Note that running with
an FPM requires HAWC2 >= 12.9 or HAWCStab2 >= 2.16. 

Both blade models feature torsional deflections and are therefore
equivalent to a BeamDyn model. To make an ElastoDyn equivalent, 
increase the torsional stiffness (shear modulus) by several orders
of magnitude.
