# IEA-15-240-RWT
15MW reference wind turbine repository developed in conjunction with IEA Wind

This repository is the home for the dissemination of files related to the in-development IEA 15MW Offshore Reference Turbine.  Community use and feedback on the provided data, and turbine design more generally, are encouraged.  Please be aware that the turbine design and model development is ongoing and subject to regular changes and corrections.

 
## Upcoming features:

* Design refinements: nacelle bedplate, yaw bearing and drive, airfoil polar data
* OpenFAST features: yaw system spring and damping constant
* Industry review of monopile design

## Update 11/20/2019:

* Generator and nacelle redesign based on industry feedback
* Monopile and tower redesign for updated RNA mass, minimum rotor speed increased to 5 RPM to avoid tower 3P excitation
* Updated contoller with shutdown and minimum rotor speed, source code available at: https://github.com/NREL/ROSCO
* Updated tower documentation and new spreadsheet with RNA high level modeling (OpenFAST, Hawc2, and WISDEM) properties (mass, intertia, geometry)

## Update 10/17/2019:

* Adding some documentation on generator design
* Fixed mismatch between tower and monopile

## Update 10/15/2019:

* First upload of monopile foundation in OpenFAST model
* First upload of yaml file for use with multiple repos

## Update 10/13/2019:

* OpenFAST controller in place thanks to Nikhar Abbas
* Include next version of OpenFAST files to fix twist and spar cap rotation bug

## Update 10/9/2019:

* Switching over to permanent GitHub home
* Addition of OpenFAST controller from Nikhar Abbas

## Update 10/6/2019:

* Corrections to tower properties and mode shapes
* Corrections to RNA masses and inertias
* Yaw and drivetrain stiffness and damping added
