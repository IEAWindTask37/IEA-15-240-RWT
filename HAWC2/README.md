# HAWC2 models

## Folders

The HAWC2 models are organized into folders:  
 * **IEA-15-240-RWT**: Contains partial htc files that are
   loaded/reused in the different models.  
 * **IEA-15-240-RWT-Onshore**: The model is fixed
   in translation and rotation at the bottom of the tower,
   15 m above the mean water level.
 * **IEA-15-240-RWT-Monopile**: IEA 15 MW in 30-m water depth
   with a monopile foundation. Same tower as Onshore model.
 * **IEA-15-240-RWT-UMaineSemi**: IEA 15 MW mounted on a
   semisubmersible foundation. Different tower than
   Onshore/monopile models.

## HTC file(s)

Only one htc file is provided in each model folder: a 100s,
steady-wind simulation with constant shear, no FPM (equivalent
to a BeamDyn model). See "Blade Models" below for more
information on the options for blade modeling.

For convenience, we have provided helper Python scripts in 
`scripts/make_htc.py` that make multiple types of htc files
from the base file, including HAWCStab2 files. The Python
scripts require Python
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

## Tower properties

The Onshore and Monopile models have the same tower. The UMaine
semisub model has a different tower, which was redesigned to
avoid resonance on the floating platform.

Each HAWC2 model has three sets of tower structural properties:
(1) a fully flexible tower, (2) a torsionally rigid tower, and
(3) a fully rigid tower. OpenFAST currently cannot model tower
torsion, so if you want to compare to OpenFAST results you should
use the second option.

The tower models use a single body, i.e., assume linear deflections.
This is to allow easy comparisons with OpenFAST, but it should be
noted that towers of this size have nonlinear deflections.
