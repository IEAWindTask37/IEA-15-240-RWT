# HAWC2 models

* **FPM**: Fully populated matrix. This is HAWC2's Timoshenko model with a 6x6 stiffness matrix. 
  Can be considered as HAWC2's equivalent of BeamDyn.

* **NoFPM**: Same properties as FPM but modelled using HAWC2's classical Timoshenko beam.

* **NoFPM_notorsion**: Similar to NoFPM, except torsion is removed by setting G arbitrarily high.
  Can be considered as HAWC2's equivalent of ElastoDyn.

The FPM and NoFPM models have the same controller parameters.
The controller parameters for the NoFPM_notorsion model were retuned using HAWCStab2.