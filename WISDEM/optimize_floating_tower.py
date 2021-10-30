#!/usr/bin/env python3
import os
from wisdem import run_wisdem
import wisdem.postprocessing.compare_designs as compare_designs 
import wisdem.postprocessing.wisdem_get as getter


# File management
thisdir = os.path.dirname(os.path.realpath(__file__))
ontology_dir = os.path.join(os.path.dirname(thisdir), "WT_Ontology")
fname_wt_input = os.path.join(ontology_dir, "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling = os.path.join(thisdir, "modeling_options_floating_tower.yaml")
fname_analysis_noopt = os.path.join(thisdir, "analysis_options.yaml")
fname_analysis_opt = os.path.join(thisdir, "analysis_options_floating_tower.yaml")
folder_output = os.path.join(thisdir, "outputs")

# Don't enforce a hub-height, just use tower description
overridden_values={'configuration.hub_height_user':0.0}

# Run WISDEM tower-monopile optimization
prob, modeling_options, analysis_noopt = run_wisdem(fname_wt_input, fname_modeling, fname_analysis_noopt,
                                                    overridden_values=overridden_values)
wt_opt, modeling_options, analysis_opt = run_wisdem(fname_wt_input, fname_modeling, fname_analysis_opt,
                                                    overridden_values=overridden_values)

# Produce standard comparison plots
compare_designs.run([prob, wt_opt], ['Before','After'], modeling_options, analysis_opt)

# print results from the analysis or optimization
z = 0.5 * (wt_opt["floatingse.tower.z_full"][:-1] + wt_opt["floatingse.tower.z_full"][1:])
print("zs =", wt_opt["floatingse.tower.z_full"])
print("ds =", wt_opt["floatingse.tower.d_full"])
print("ts =", wt_opt["floatingse.tower.t_full"])
print("mass (kg) =", getter.get_tower_mass(wt_opt))
print("cg (m) =", getter.get_tower_cg(wt_opt)[-1])
print("d:t constraint =", wt_opt["floatingse.tower.constr_d_to_t"])
print("taper ratio constraint =", wt_opt["floatingse.tower.constr_taper"])
print("freq (Hz) =", getter.get_tower_freqs(wt_opt))
print("Fore-aft mode shapes =", wt_opt["floatingse.tower_fore_aft_modes"])
print("Side-side mode shapes =", wt_opt["floatingse.tower_side_side_modes"])
ind = wt_opt["floatingse.constr_tower_stress"] > -999
print("stress1 =", wt_opt["floatingse.constr_tower_stress"][ind])
print("GL buckling =", wt_opt["floatingse.constr_tower_global_buckling"][ind])
print("Shell buckling =", wt_opt["floatingse.constr_tower_shell_buckling"][ind])

'''
# print results from the analysis or optimization
z = 0.5 * (wt_opt["towerse.z_full"][:-1] + wt_opt["towerse.z_full"][1:])
print("zs =", wt_opt["towerse.z_full"])
print("ds =", wt_opt["towerse.d_full"])
print("ts =", wt_opt["towerse.t_full"])
print("mass (kg) =", wt_opt["towerse.tower_mass"])
print("cg (m) =", wt_opt["towerse.tower_center_of_mass"])
print("d:t constraint =", wt_opt["towerse.constr_d_to_t"])
print("taper ratio constraint =", wt_opt["towerse.constr_taper"])
print("\nwind: ", wt_opt["towerse.env.Uref"])
print("freq (Hz) =", wt_opt["towerse.tower.structural_frequencies"])
print("Fore-aft mode shapes =", wt_opt["towerse.tower.fore_aft_modes"])
print("Side-side mode shapes =", wt_opt["towerse.tower.side_side_modes"])
print("top_deflection1 (m) =", wt_opt["towerse.tower.top_deflection"])
print("Tower base forces1 (N) =", wt_opt["towerse.tower.turbine_F"])
print("Tower base moments1 (Nm) =", wt_opt["towerse.tower.turbine_M"])
print("stress1 =", wt_opt["towerse.post.constr_stress"].flatten())
print("GL buckling =", wt_opt["towerse.post.constr_global_buckling"].flatten())
print("Shell buckling =", wt_opt["towerse.post.constr_shell_buckling"].flatten())
'''
