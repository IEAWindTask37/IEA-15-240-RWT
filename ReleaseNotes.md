# IEA Wind 15-MW Release Notes

## v 1.1

This update to the IEA Wind 15-MW Reference Wind Turbine attempts to address some of the modeling issues that have been pointed out by users.  Larger redesign requests, such as a smaller diameter monopile, or higher specific power for the floating turbine, are not included as that would require a more significant allocation of resources (see the [Wiki FAQ](https://github.com/IEAWindTask37/IEA-15-240-RWT/wiki/Frequently-Asked-Questions-(FAQ)) for some extended answers to those topics).  

**Major Changes**

 * Many of the modeling tools have undergone changes and updates, so the files here have been updated to keep pace with API changes.
 
   - Changes WISDEM has undergone perhaps the most extensive overhaul.  It now uses the YAML-based ontology files from the [WindIO](https://github.com/IEAWindTask37/windIO) project as the chief input, and is also more tightly integrated with OpenFAST and ROSCO.  More detailed changes described below.
   
   - OpenFAST has seen a couple of releases since the 15-MW RWT input files were created.  Additionally, the original input deck also had a number of errors and inconsistencies.  The integration with WISDEM and, by extension, the WindIO ontology files has help to rectify these problems.  Finally, a set of input files that works with OLAF, the OpenFAST vortex particle aerodynamics code has been included.
   
   - ROSCO, which provides the controller, was a new tool when the 15-MW RWT was first released and has since seen numerous bug fixes and new features.  More details described below and at the [ROSCO repository](https://github.com/nrel/rosco).

 * The ontology YAML-files have been expanded to include descriptions of the entire turbine.  This follows the progress made in the parallel IEA Task 37 [WindIO](https://github.com/IEAWindTask37/windIO) project.  Note that the YAML descriptions for some of these components will likely still evolve, especially the nacelle components, as settling on a communal parameterization takes time and numerous iterations.

 * The nacelle component mass properties and dimensions were approximate in the report and then inconsistently captured between the various model input decks.  Furthermore Table 5-1 in the report had values that could do be replicated and did not add up (called out in [Issue #30](https://github.com/IEAWindTask37/IEA-15-240-RWT/issues/30).  With the new capabilities in WISDEM's DrivetrainSE model, this has been rectified and the nacelle mass properties tab in the IEA-15-240-RWT_tabular.xlsx file.
 
 * The composite material properties used in the blade has been updated to be more consistent with modern materials and manufacturing techniques.  As a result, the blade mass has grown, even though the geometry is the same.  Originally WISDEM (via the PreComp tool) estimated blade mass as approximately 65t, which was stated in the report.  With the updated properties that estimate is now 73t.  Ideally we would re-design the blades around these new properties, but that is a more significant undertaking.  This was also called out in the [Wiki FAQ](https://github.com/IEAWindTask37/IEA-15-240-RWT/wiki/Frequently-Asked-Questions-(FAQ)).
 
 * Trailing edge reinforcement was mistakenly added too close to root and tip of blade in the first release, so this has been corrected.
Blade mass change due to material properties


**Detailed OpenFAST changes**

For the monopile variant:

 * InflowWind:
 
    - RefLength corrected to rotor diameter
    - API changes to be consistent with OpenFAST v2.6
	
 * AeroDyn:
 
    - Blade mass and stiffness properties have been updated per the updates to the material properties.
    - The blade mode shapes for ElastoDyn were incorrect in multiple ways.  In the initial release, the flap and edge modes were both the first bending mode (as captured in [Issue #37](https://github.com/IEAWindTask37/IEA-15-240-RWT/issues/37).  Furthermore, there was a slight error in the polynomial fitting routine that led to some non-physical trends in ElastoDyn in flexible blades.  Both of these issues have been corrected.
    - API changes to be consistent with OpenFAST v2.6
	
 * ElastoDyn:
 
    - Zeroed initial platform heave, PtfmHeave
    - Made small changes to (TipRad, HubRad, Overhang, Twr2Shft, TowerHt) to match ontology values
    - NacCMxn/yn/zn updated based on bottom-up WISDEM DrivetrainSE calculation and tracking
    - Nacelle component masses and MofI calculated through WISDEM DrivetrainSE.  Values for HubMass, HubIner, GenIner, NacMass, NacYIner, YawBrMass, DTTorSpr, and DTTorDmp all updated accordingly.
    - PtfmYIner zeroed out because transition piece is accounted for in SubDyn
    - Tower modes shape coefficients were updated with corrections to the polynomial fit routines
	
 * HydroDyn
 
    - Made small changes to (WaveHs, WaveTp) to match ontology values
    - Monopile geometry specification is the same as SubDyn and the ontology file, with more accurate thickness profile
	
 * SubDyn:
 
    - Transition piece added as a concentrated mass at top of monopile
	
 * ServoDyn:
 
    - Small change to GenEff to match bottom-up estimate (from 96.55 to 95.756)
    - Default GenModel switched to 1 (simple)
    - Zeroed HSS brake data since that has not been specified in the design
    - YawSpr set to torsional stiffness of tower and YawDamp set for 1% damping ratio for first tower torsional mode
    - YawManRat changed from 2 deg/s to 0.25 deg/s due to size of RNA
    - Turned off DLL_DT and DLL_Ramp
	
 * MoorDyn:
 
    - API changes to be consistent with OpenFAST v2.6

For the UMaine Volturn-S floating variant:

 * All of the changes associated with the monopile variant
 
 * RotSpeed set to 7.55, the rated speed

Also added input files that invoke OLAF, the OpenFAST vortex particle method for aerodynamics (monopile variant only)


**Detailed HAWC2 changes**

(Jenni / DTU to-do items)


**Detailed WISDEM changes**

Three WISDEM scripts remain:

 * `run_model.py` : WISDEM analysis (no design optimization) of the IEA Wind 15-MW RWT.  Uses the WindIO ontology file directly as input, along with `modeling_options.yaml` and `analysis_options.yaml`. Script also includes automated plotting of blade geometry, rotor performance, and tower geometry.  The tabular data Excel sheet is also generated.
 
 * `optimize_monopile_tower.py` : Script that can be used to optimize the tower and monopile diameter and thickness schedule.  Uses the WindIO ontology file directly as input, along with `modeling_options_monopile.yaml` and `analysis_options_monopile.yaml`.
 
 * `optimize_generator.py` : Script that can be used to optimize the permanent magnet synchronous generator.
 
For more information about WISDEM and how these scripts can be run, please see the WISDEM [documentation](https://wisdem.readthedocs.io) and [repository](https://github.com/WISDEM/WISDEM).


**Detailed ROSCO changes**

(just point folks to the ROSCO repo and commit logs?)


## v 1.0

 * Jan 2021: Officially tagged the v1.0 release, although there are still some outstanding issues
 
 * Apr-Nov, 2020: Series of minor fixes in model input files (see closed pull requests for details)
 
 * March 30, 2020: Finalization of OpenFAST and HAWC2 models in conjuction with release of report and media pieces
 

