#!/usr/bin/env python3
import os
from wisdem import run_wisdem
import wisdem.postprocessing.compare_designs as compare_designs 


# File management
thisdir = os.path.dirname(os.path.realpath(__file__))
ontology_dir = os.path.join(os.path.dirname(thisdir), "WT_Ontology")
fname_wt_input = os.path.join(ontology_dir, "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling = os.path.join(thisdir, "modeling_options_floating_tower.yaml")
fname_analysis_noopt = os.path.join(thisdir, "analysis_options.yaml")
fname_analysis_opt = os.path.join(thisdir, "analysis_options_floating_tower.yaml")
folder_output = os.path.join(thisdir, "outputs")

# Run WISDEM tower-monopile optimization
prob, modeling_options, analysis_noopt = run_wisdem(fname_wt_input, fname_modeling, fname_analysis_noopt)
wt_opt, modeling_options, analysis_opt = run_wisdem(fname_wt_input, fname_modeling, fname_analysis_opt)

# Produce standard comparison plots
compare_designs.run([prob, wt_opt], ['Before','After'], modeling_options, analysis_opt)

# print results from the analysis or optimization
z = 0.5 * (wt_opt["floatingse.tower.z_full"][:-1] + wt_opt["floatingse.tower.z_full"][1:])
print("zs =", wt_opt["floatingse.tower.z_full"])
print("ds =", wt_opt["floatingse.tower.d_full"])
print("ts =", wt_opt["floatingse.tower.t_full"])
print("mass (kg) =", wt_opt["floatingse.tower.tower_mass"])
print("cg (m) =", wt_opt["floatingse.tower.center_of_mass"])
print("d:t constraint =", wt_opt["floatingse.tower.constr_d_to_t"])
print("taper ratio constraint =", wt_opt["floatingse.tower.constr_taper"])
print("freq (Hz) =", wt_opt["floatingse.tower.tower_frequencies"])
print("Fore-aft mode shapes =", wt_opt["floatingse.tower.fore_aft_modes"])
print("Side-side mode shapes =", wt_opt["floatingse.tower.side_side_modes"])
print("top_deflection1 (m) =", wt_opt["floatingse.tower.top_deflection"])
print("stress1 =", wt_opt["floatingse.tower_stress"])
print("GL buckling =", wt_opt["floatingse.tower_global_buckling"])
print("Shell buckling =", wt_opt["floatingse.tower_shell_buckling"])
