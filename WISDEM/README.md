# IEA Wind 15-MW RWT: WISDEM Model

Provided WISDEM scripts:

 * `run_model.py` : WISDEM analysis (no design optimization) of the IEA Wind 15-MW RWT.  Uses the WindIO ontology file directly as input, along with `modeling_options.yaml` and `analysis_options.yaml`. Script also includes automated plotting of blade geometry, rotor performance, and tower geometry.  The tabular data Excel sheet is also generated.
 
 * `optimize_monopile_tower.py` : Script that can be used to optimize the tower and monopile diameter and thickness schedule.  Uses the WindIO ontology file directly as input, along with `modeling_options_monopile.yaml` and `analysis_options_monopile.yaml`.
 
 * `optimize_generator.py` : Script that can be used to optimize the permanent magnet synchronous generator.
 
For more information about WISDEM and how these scripts can be run, please see the WISDEM [documentation](https://wisdem.readthedocs.io) and [repository](https://github.com/WISDEM/WISDEM).

