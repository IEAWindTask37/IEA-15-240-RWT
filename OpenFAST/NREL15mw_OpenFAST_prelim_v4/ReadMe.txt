This is the preliminary in-progress OpenFAST model for the yet to be released NREL 15MW Offshore Reference Turbine.

ver. 4.0
10/1/2019

Please note that this model has not been thoroughly tested yet!!

Due to the preliminary status of this model, a number of subsystems have yet to be included.  Current Limitations:
 - Controller not yet included
 - Drivetrain torsion stiffness and damping not updated, set DrTrDOF = False
 - Yaw stiffness and damping not updated, set YawDOF = False
 - Monopile hydrodyn and subdyn files not yet included, note that the tower still starts at 15m above the ground
 - Airfoils interpolated based on spanwise thickness, updated version will include a non-interpolated blade option


