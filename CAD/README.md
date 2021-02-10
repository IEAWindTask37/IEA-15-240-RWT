# IEA Wind 15-MW RWT: CAD Files
 
The CAD files provided here are:

 * `IEA-15-240-RWT_Solidworks.zip` : Original Solidworks CAD model of the full monopile variant.  Drivetrain geometry and material properties are likely incorrect.  Blade material properties may be out of date.
 
 * `IEA-15-240-RWT_blade.stl` : High resolution blade outer geometry STL file produced by the  [WindIO2CAD](https://github.com/IEAWindTask37/windio2cad) project.  No internal structure provided.

 * `IEA-15-240-RWT_Ansys` : Quick attempt to provide structural analysis files for the blade.  Feedback from users has been that this is not accurate or sufficient for most analysis needs.  Kept for archival purposes.

 * `Generator_detail.zip` : Original Solidworks assembly for the permanent magnet synchronous generator.
  
For users that are interested in CAD files for the purposes of CFD simulations, and therefore require detailed outer shape geometry only, we recommend using the [WindIO2CAD](https://github.com/IEAWindTask37/windio2cad) project that can generate STL-files directly from the WindIO ontology via [OpenSCAD](https://www.openscad.org/).

For users that would like to conduct high-fidelity structural analysis of the IEA Wind 15-MW RWT, the files here are likely inadequate for your needs.  IEA Wind Task 37 is aware some community efforts to generate Ansys models of the blades and support structure, but nothing is yet ready to share.

