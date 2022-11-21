
**NOTICE**

This directory includes WAMIT output files for hydrostatics, radiation, and both first- and second-order wave excitation.  The first-order excitation file (.3 extension) supports all wave headings, whereas the second-order excitation files (.12d and .12s extensions) are only provided for the 0-degree incident wave heading.  The HydroDyn flags, WvDiffQTF and WvSumQTF, can be used to activate or deactivate the inclusion of the second-order effects files, depending on the desired wave loading.  The WAMIT out-files (labeled as .out.wamit) are also included.

The WAMIT input files are included in two subdirectories, one for first order and one for second order. The first-order setup (used for the .1 and .3 files) include zero- and infinite-frequency radiation entries and include wave excitation at all headings in 10 degree increments. The second-order settings (used for the .12s and .12d files) use a reduced frequency range and only the zero degree wave excitation heading to reduce computational expense.

Note that the center of gravity is moved to the origin in the first-order WAMIT input file to remove the contributions from gravity to the hydrostatic stiffness in *.hst. The moment from gravity is added separately by OpenFAST. 
