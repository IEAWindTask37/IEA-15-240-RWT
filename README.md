# IEA-15-240-RWT
This repository contains the model data for the 15 MW offshore reference turbine developed within IEA Wind Task 37.

The documentation for the turbine is accessable here: tbd

Data in this repository includes:
* Documentation, including tabular data used in the figures from the technical report
* OpenFAST aeroelastic model inputs
* HAWC2 aeroelastic model inputs
* WISDEM optimization files
* Wind turbine ontology .yaml files
* CAD modeling of turbine in SolidWorks and an Ansys FEA model of the blades

Note that the OpenFAST model uses the NREL's [Reference OpenSource Controller (ROSCO)](https://github.com/nrel/rosco).  Users will need to compile the dynamic library following the ROSCO install instructions and need to provide relavent path to the dynamic library within ServoDyn.  [Tools](https://github.com/NREL/ROSCO_toolbox) also exist for automatically retuning the controller, updating the "Cp_Ct_Cq.IEA15MW.txt" and "DISCON.IN" files that are inputs to ROSCO.  Please use OpenFAST-v2.2.0 or later, the OpenFAST model has not been tested for backwards compatability.

If you use this model in your research or publications, please cite as:

    @techreport{IEA15MW_ORWT,
    author = {Evan Gaertner and Jennifer Rinker and Latha Sethuraman and Frederik Zahle and Benjamin Anderson and Garrett Barter and Nikhar Abbas and Fanzhong Meng and Pietro Bortolotti and Witold Skrzypinski and George Scott and Roland Feil and Henrik Bredsmoe and Katherine Dykes and Matt Sheilds and Christopher Allen and Anthony Viselli},
    Howpublished = {NREL/TP-75698},
    institution = {International Energy Agency},
    title = {{Definition of the IEA 15 MW Offshore Reference Wind Turbine}},
    URL = {tbd},
    Year = {}
    }

For questions, contact Evan Gaertner (evan.gaertner@nrel.gov).  The technical report lists the contributions of individual authors, which may provide a more relevant point of contact.