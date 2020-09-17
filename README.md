# IEA-15-240-RWT
This repository contains the model data for the 15-MW offshore reference turbine developed within IEA Wind Task 37.

The documentation for the turbine is accessible here: https://www.nrel.gov/docs/fy20osti/75698.pdf
and the semisubmersible floating support structure is documented here: https://www.nrel.gov/docs/fy20osti/76773.pdf

Data in this repository includes:
* Documentation, including tabular data used in the figures from the technical report
* OpenFAST aeroelastic model inputs
* HAWC2 aeroelastic model inputs
* WISDEM optimization files
* Wind turbine ontology .yaml files
* CAD modeling of turbine in SolidWorks and an Ansys FEA model of the blades

## Requirements

Requirements for using the OpenFAST model:
* The OpenFAST input files are compatable with OpenFAST-v2.3+.  This can be compiled [from source here](https://github.com/OpenFAST/openfast.git) or precompiled Windows binaries are [available for download](https://github.com/OpenFAST/openfast/releases/latest/download/windows_openfast_binaries.zip). More information on installing and running OpenFAST is available in [OpenFAST documention](https://openfast.readthedocs.io/en/master/). 
* NREL's Reference OpenSource Controller (ROSCO) is required.  This can be compile [from source here](https://github.com/nrel/rosco) or the precompiled Windows .dll is [available for download](https://github.com/NREL/ROSCO/releases/tag/v2.0.1).

## Design Updates

If the IEA Wind Task 37 authors deems it appropriate, design updates will be uploaded to the repository and tagged with version numbers appropriately.  This will ensure that any design oversights are addressed and that the input files stay current with model updates.

We also encourage the broader wind community to submit design updates either through Pull Requests or by reaching out to the authors.  Community contributions that we are aware of include:
* [OrcaFlex implementation](https://github.com/IEAWindTask37/IEA-15-240-RWT/wiki/Frequently-Asked-Questions-(FAQ)#is-orcaflex-supported)


## Questions

Before reaching out to NREL or DTU authors with questions on the model or reports, please see our frequently asked questions (FAQ) on our [Github Wiki](https://github.com/IEAWindTask37/IEA-15-240-RWT/wiki/Frequently-Asked-Questions-(FAQ)).

If the FAQ does not address your need, please create a new Issue on this repository so that the dialogue is archived for others that might have similar questions. You can also reach out to the authors directly if that is your preference.

## Citation

If you use this model in your research or publications, please cite as:

    @techreport{IEA15MW_ORWT,
    author = {Evan Gaertner and Jennifer Rinker and Latha Sethuraman and Frederik Zahle and Benjamin Anderson and Garrett Barter and Nikhar Abbas and Fanzhong Meng and Pietro Bortolotti and Witold Skrzypinski and George Scott and Roland Feil and Henrik Bredmose and Katherine Dykes and Matt Sheilds and Christopher Allen and Anthony Viselli},
    Howpublished = {NREL/TP-75698},
    institution = {International Energy Agency},
    title = {{Definition of the IEA 15 MW Offshore Reference Wind Turbine}},
    URL = {https://www.nrel.gov/docs/fy20osti/75698.pdf},
    Year = {2020}
    }

For questions, contact Garrett Barter (garrett.barter@nrel.gov).  The technical report lists the contributions of individual authors, which may provide a more relevant point of contact.
