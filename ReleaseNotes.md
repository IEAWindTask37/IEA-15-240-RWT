March 30: Finalization of OpenFAST and HAWC2 models in conjuction with release of report and media pieces
Apr-Nov: Series of minor fixes in model input files (see closed pull requests for details)

Nacelle mass property breakdown replaces Table X-X in report
Monopile-tower redesign with corrected spring-stiffness boundary conditions
Rely on WindIO for ontology schema
TE reinfocement was mistakenly added too close to root and tip of blade
Nacelle mass properties cleaned up and self-consistent.  Tabular replaces Report Ttaable XX
Blade mass change due to material properties

Account for slight OpenFAST API changes to AeroDyn15, InflowWind, MoorDyn
Blade mass and elastic properties updated with more accurate and current composite material properties (see Wiki)
Blade mode shapes
InflowWind:
- RefLength corrected to rotor diameter
ElastoDyn:
- Zeroed initial platform heave
- Made small changes to (TipRad, HubRad, Overhang, Twr2Shft, TowerHt) to match ontology values
- NacCMxn/yn/zn updated based on bottom-up WISDEM calculation and tracking
- Nacelle component masses and MofI are now bottom calculated instead of guesses.  Values for HubMass, HubIner, GenIner, NacMass, NacYIner, YawBrMass all updated
- DTTorSpr and DTTorDmp updated similarly 
- PtfmYIner zero-ed out because transition piece is accounted for in SubDyn
- Tower modes shape coefficients were updated with corrections to the polynomial fit routines
HydroDyn
- Made small changes to (WaveHs, WaveTp) to match ontology values
- Monopile geometry matches that in SubDyn and the ontology file, with more accurate thickness changes
SubDyn:
- Transition piece added as a concentrated mass at top of monopile
ServoDyn:
- Small change to GenEff to match bottom-up estimate (96.55 to 95.756)
- Default GenModel switched to 1 (simple)
- Zeroed HSS brake data as that was not part of design
- YawSpr set to torsional stiffness of tower and YawDamp set for 1% damping ratio for first tower torsional mode
- YawManRat changed from 2 deg/s to 0.25 deg/s due to size of RNA
- Turned off DLL_DT and DLL_Ramp

UMaine SEmi version (monopile changes plus):
- RotSpeed set to 7.55, the rated speed

Added OLAF version for users that would like to try vortex particle aerodynamics
