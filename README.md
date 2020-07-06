# IEA-15-240-RWT
This repository contains the model data for the 15 MW offshore reference turbine developed within IEA Wind Task 37.

The documentation for the turbine is accessable here: https://www.nrel.gov/docs/fy20osti/75698.pdf

Data in this repository includes:
* Documentation, including tabular data used in the figures from the technical report
* OpenFAST aeroelastic model inputs (compatable with OpenFAST-v2.3+) available: )
* HAWC2 aeroelastic model inputs
* WISDEM optimization files
* Wind turbine ontology .yaml files
* CAD modeling of turbine in SolidWorks and an Ansys FEA model of the blades

Requirements for using the OpenFAST model:
* The OpenFAST input files are compatable with OpenFAST-v2.3+.  This can be compiled [from source here](https://github.com/OpenFAST/openfast.git) or precompiled Windows binaries are [available for download](https://github.com/OpenFAST/openfast/releases/latest/download/windows_openfast_binaries.zip). More information on installing and running OpenFAST is available in [OpenFAST documention](https://openfast.readthedocs.io/en/master/). 
* NREL's Reference OpenSource Controller (ROSCO) is required.  This can be compile [from source here](https://github.com/nrel/rosco) or precompiled Windows binaries are [available for download](https://github.com/NREL/ROSCO/releases/tag/latest).

If you use this model in your research or publications, please cite as:

    @techreport{IEA15MW_ORWT,
    author = {Evan Gaertner and Jennifer Rinker and Latha Sethuraman and Frederik Zahle and Benjamin Anderson and Garrett Barter and Nikhar Abbas and Fanzhong Meng and Pietro Bortolotti and Witold Skrzypinski and George Scott and Roland Feil and Henrik Bredmose and Katherine Dykes and Matt Sheilds and Christopher Allen and Anthony Viselli},
    Howpublished = {NREL/TP-75698},
    institution = {International Energy Agency},
    title = {{Definition of the IEA 15 MW Offshore Reference Wind Turbine}},
    URL = {https://www.nrel.gov/docs/fy20osti/75698.pdf},
    Year = {2020}
    }

For questions, contact Evan Gaertner (evan.gaertner@nrel.gov).  The technical report lists the contributions of individual authors, which may provide a more relevant point of contact.
