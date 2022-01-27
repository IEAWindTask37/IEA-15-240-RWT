#!/usr/bin/env python3
import os
from wisdem import run_wisdem
import wisdem.postprocessing.compare_designs as compare_designs 


# File management
thisdir = os.path.dirname(os.path.realpath(__file__))
ontology_dir = os.path.join(os.path.dirname(thisdir), "WT_Ontology")
fname_wt_input = os.path.join(ontology_dir, "IEA-15-240-RWT.yaml")
fname_modeling = os.path.join(thisdir, "modeling_options_monopile.yaml")
fname_analysis_noopt = os.path.join(thisdir, "analysis_options.yaml")
fname_analysis_opt = os.path.join(thisdir, "analysis_options_monopile.yaml")
folder_output = os.path.join(thisdir, "outputs")

# Run WISDEM tower-monopile optimization
prob, modeling_options, analysis_noopt = run_wisdem(fname_wt_input, fname_modeling, fname_analysis_noopt)
wt_opt, modeling_options, analysis_opt = run_wisdem(fname_wt_input, fname_modeling, fname_analysis_opt)

# Produce standard comparison plots
compare_designs.run([prob, wt_opt], ['Before','After'], modeling_options, analysis_opt)


# print results from the analysis or optimization
print("\n\nTower-monopile z-pts =", wt_opt["towerse.z_param"])
print("Tower diameter =", wt_opt["towerse.tower_outer_diameter"])
print("Tower thickness =", wt_opt["towerse.tower_wall_thickness"])
print("Tower mass (kg) =", wt_opt["towerse.tower_mass"])
print("Monopile diameter =", wt_opt["fixedse.monopile_outer_diameter"])
print("Monopile thickness =", wt_opt["fixedse.monopile_wall_thickness"])
print("Monopile mass (kg) =", wt_opt["fixedse.monopile_mass"])
print("Total mass (kg) =", wt_opt["fixedse.structural_mass"])

print("\nTower Fore-aft freq (Hz) =", wt_opt["towerse.tower.fore_aft_freqs"])
print("Tower Fore-aft mode shapes =", wt_opt["towerse.tower.fore_aft_modes"])
print("Tower Side-side freq (Hz) =", wt_opt["towerse.tower.side_side_freqs"])
print("Tower Side-side mode shapes =", wt_opt["towerse.tower.side_side_modes"])
print("Monopile Fore-aft freq (Hz) =", wt_opt["fixedse.monopile.fore_aft_freqs"])
print("Monopile Fore-aft mode shapes =", wt_opt["fixedse.monopile.fore_aft_modes"])
print("Monopile Side-side freq (Hz) =", wt_opt["fixedse.monopile.side_side_freqs"])
print("Monopile Side-side mode shapes =", wt_opt["fixedse.monopile.side_side_modes"])

print("\nwind: ", wt_opt["towerse.env.Uref"])
print("Tower top_deflection (m) =", wt_opt["towerse.tower.top_deflection"])
print("Tower base forces (N) =", wt_opt["towerse.tower.turbine_F"])
print("Tower base moments (Nm) =", wt_opt["towerse.tower.turbine_M"])
print("Tower Constraint z-pts =", wt_opt["towerse.z_full"])
print("Tower stress =", wt_opt["towerse.post.constr_stress"].flatten())
print("Tower GL buckling =", wt_opt["towerse.post.constr_global_buckling"].flatten())
print("Tower Shell buckling =", wt_opt["towerse.post.constr_shell_buckling"].flatten())
print("Tower taper ratio constraint =", wt_opt["towerse.constr_taper"])
print("Monopile top_deflection (m) =", wt_opt["fixedse.monopile.top_deflection"])
print("Mudline forces (N) =", wt_opt["fixedse.monopile.mudline_F"])
print("Mudline moments (Nm) =", wt_opt["fixedse.monopile.mudline_M"])
print("Monopile Constraint z-pts =", wt_opt["fixedse.z_full"])
print("Monopile stress =", wt_opt["fixedse.post.constr_stress"].flatten())
print("Monopile GL buckling =", wt_opt["fixedse.post.constr_global_buckling"].flatten())
print("Monopile Shell buckling =", wt_opt["fixedse.post.constr_shell_buckling"].flatten())
print("Monopile taper ratio constraint =", wt_opt["fixedse.constr_taper"])

