[![DOI](https://zenodo.org/badge/213679527.svg)](https://zenodo.org/badge/latestdoi/213679527)

# IEA-15-240-RWT v1.1 (Revised February 2022)
This repository contains the model data for the 15-MW offshore reference turbine developed within IEA Wind Task 37.

The documentation for the turbine is accessible here: https://www.nrel.gov/docs/fy20osti/75698.pdf
and the semisubmersible floating support structure is documented here: https://www.nrel.gov/docs/fy20osti/76773.pdf

Data in this repository includes:
* Documentation, including tabular data used in the figures from the technical report
* OpenFAST aeroelastic model inputs
* HAWC2 aeroelastic model inputs
* WISDEM optimization files
* Wind turbine ontology .yaml files
* CAD modeling of turbine where available

## Requirements

*OpenFAST*:
* The OpenFAST input files are compatable with OpenFAST-v3.0.  This can be compiled [from source here](https://github.com/OpenFAST/openfast.git) or precompiled Windows binaries are [available for download](https://github.com/OpenFAST/openfast/releases/latest/download/windows_openfast_binaries.zip). More information on installing and running OpenFAST is available in [OpenFAST documention](https://openfast.readthedocs.io/en/master/).
* NREL's Reference OpenSource Controller (ROSCO) is required.  This can be compiled [from source here](https://github.com/nrel/rosco) or precompiled binaries for all platforms are [available for download](https://github.com/NREL/ROSCO/releases/).

*HAWC2*:
* HAWC2 can be acquired from its [homepage](https://www.hawc2.dk/).  The DTU Basic Controller can be obtained from its [repository](https://gitlab.windenergy.dtu.dk/OpenLAC/BasicDTUController).

*WISDEM*:
 * WISDEM can be installed from its Github [repository](https://github.com/WISDEM/WISDEM).
 * See the [documentation](https://wisdem.readthedocs.io) for installation and usage guides.


## Design Updates

The IEA Wind Task 37 authors endeavor to keep the model input decks current with the latest releases and API changes.  Errors and other issues pointed out by the community are also addressed to the extent that available resources make that possible.  See the [Release Notes](blob/master/ReleaseNotes.md) for a detailed description of changes.

We also encourage the broader wind community to submit design updates by forking the repository and letting us know of your design customatization.  Community contributions that we are aware of include:
* [Bladed model](https://github.com/IEAWindTask37/IEA-15-240-RWT/wiki/Frequently-Asked-Questions-(FAQ)#is-bladed-supported) implemented by DNVGL, contact [William Collier](mailto:william.collier@dnv.com)
* [OrcaFlex model](https://github.com/IEAWindTask37/IEA-15-240-RWT/wiki/Frequently-Asked-Questions-(FAQ)#is-orcaflex-supported) implemented by Orcina, contact [Alan Ross](mailto:Alan.Ross@orcina.com)
* [Detailed rotor redesign](https://data.bris.ac.uk/data/dataset/3jrb4mejp9vfd2qb3s7dreymr1) from University of Bristol described in a TORQUE 2022 [paper](https://iopscience.iop.org/article/10.1088/1742-6596/2265/3/032029/pdf), contact [Peter Greaves](mailto:peter.greaves@ore.catapult.org.uk)
* [Jacket support structure](https://github.com/mmrocze2/IEA-15-240-RWT) The DEME Group created a 3-legged jacket for a 50m water depth, contact [Maciej Mroczek](mailto:Mroczek.Maciej@deme-group.com)

## Citations

For a list of academic papers that use or cite this turbine, please see [here (fixed-bottom)](https://scholar.google.com/scholar?cites=11739673662820715884&as_sdt=4005&sciodt=0,6&hl=en) and [here (floating)](https://scholar.google.com/scholar?cites=17665986740213390479&as_sdt=4005&sciodt=0,6&hl=en).

If you use this model in your research or publications, please cite the appropriate report as:

    @techreport{IEA15MW_ORWT,
    author = {Evan Gaertner and Jennifer Rinker and Latha Sethuraman and Frederik Zahle and Benjamin Anderson and Garrett Barter and Nikhar Abbas and Fanzhong Meng and Pietro Bortolotti and Witold Skrzypinski and George Scott and Roland Feil and Henrik Bredmose and Katherine Dykes and Matt Sheilds and Christopher Allen and Anthony Viselli},
    Howpublished = {NREL/TP-75698},
    institution = {International Energy Agency},
    title = {Definition of the {IEA} 15-Megawatt Offshore Reference Wind Turbine},
    URL = {https://www.nrel.gov/docs/fy20osti/75698.pdf},
    Year = {2020}
    }

    @techreport{IEA15MW_ORWT_Floating,
    author = {Christopher Allen and Anthony Viselli and Habib Dagher and Andrew Goupee and Evan Gaertner and Nikhar Abbas and Matthew Hall and Garrett Barter},
    Howpublished = {NREL/TP-76773},
    institution = {International Energy Agency},
    title = {Definition of the {UMaine} {VolturnUS-S} Reference Platform Developed for the {IEA Wind} 15-Megawatt Offshore Reference Wind Turbine}},
    URL = {https://www.nrel.gov/docs/fy20osti/76773.pdf},
    Year = {2020}
    }

## Questions

Before reaching out to NREL or DTU authors with questions on the model or reports, please see our frequently asked questions (FAQ) on our [Github Wiki](https://github.com/IEAWindTask37/IEA-15-240-RWT/wiki/Frequently-Asked-Questions-(FAQ)) and current or prior [Issues](https://github.com/IEAWindTask37/IEA-15-240-RWT/issues).

If neither the FAQ or Issues address your need, please create a new Issue on this repository so that the dialogue is archived for others that might have similar questions. You can also reach out to the authors directly if that is your preference.  The technical report lists the contributions of individual authors if you have a specific question.  Otherwise, you can contact Garrett Barter (garrett.barter@nrel.gov).
