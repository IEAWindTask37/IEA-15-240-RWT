# Import the modules
import openmdao.api as om
import numpy as np
from wisdem.drivetrainse.generator import Generator
import wisdem.commonse.fileIO as fio

# ---

# Problem setup

# Whether or not to run optimization
opt_flag = True

# Number of points in efficiency table
n_pc = 20

# Initialize problem instance
prob = om.Problem()
prob.model = Generator(design="pmsg_outer", n_pc=n_pc)
# ---

# Pose optimization problem
if opt_flag:
    # Add optimizer and set-up problem (using user defined input on objective function)
    #prob.driver = om.ScipyOptimizeDriver()
    #prob.driver.options["optimizer"] = "SLSQP"
    #prob.driver.options["tol"] = 1e-6

    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = 'CONMIN' # 'SNOPT'

    # Specificiency target efficiency(%)
    Eta_Target = 0.955

    # Design variables
    prob.model.add_design_var("rad_ag", lower=3.0, upper=6.0)
    prob.model.add_design_var("len_s", lower=1.5, upper=2.5)
    prob.model.add_design_var("h_s", lower=0.1, upper=1.00)
    prob.model.add_design_var("p", lower=50.0, upper=100.0, ref=1e2)
    prob.model.add_design_var("h_m", lower=0.01, upper=0.2, ref=1e-2)
    prob.model.add_design_var("h_yr", lower=0.035, upper=0.22, ref=1e-2)
    prob.model.add_design_var("h_ys", lower=0.035, upper=0.22, ref=1e-2)
    prob.model.add_design_var("B_tmax", lower=1, upper=2.0)
    prob.model.add_design_var("t_r", lower=0.05, upper=0.3, ref=1e-2)
    prob.model.add_design_var("t_s", lower=0.05, upper=0.3, ref=1e-2)
    prob.model.add_design_var("h_ss", lower=0.04, upper=0.2, ref=1e-2)
    prob.model.add_design_var("h_sr", lower=0.04, upper=0.2, ref=1e-2)

    # Constraints
    prob.model.add_constraint("B_symax", lower=0.0, upper=2.0)
    prob.model.add_constraint("B_rymax", lower=0.0, upper=2.0)
    prob.model.add_constraint("b_t", lower=0.01, ref=1e-2)
    prob.model.add_constraint("B_g", lower=1.19, upper=1.2)
    prob.model.add_constraint("A_Cuscalc", lower=5.0, upper=600, ref=1e1)
    prob.model.add_constraint("K_rad", lower=0.15, upper=0.3)
    prob.model.add_constraint("Slot_aspect_ratio", lower=4.0, upper=10.0)
    prob.model.add_constraint("generator_efficiency", lower=Eta_Target, indices=[-1])
    prob.model.add_constraint("A_1", upper=110000.0, ref=1e6, indices=[-1])
    prob.model.add_constraint("T_e", lower=21.03e6, upper=21.1e6, ref=1e6)
    prob.model.add_constraint("J_actual", lower=3, upper=6, indices=[-1])
    #prob.model.add_constraint("con_uar", lower=1e-2, ref=1e-2)
    #prob.model.add_constraint("con_yar", lower=1e-2, ref=1e-2)
    #prob.model.add_constraint("con_uas", lower=1e-2, ref=1e-2)
    #prob.model.add_constraint("con_yas", lower=1e-2, ref=1e-2)

    Objective_function = "generator_cost"
    prob.model.add_objective(Objective_function, scaler=1e-5)
    # ----

# Input parameter initialization
prob.setup()

# Specify Target machine parameters
prob["machine_rating"] = 15e6
prob["shaft_rpm"] = np.linspace(5.0, 7.56, n_pc)
prob["rated_torque"] = 21030561.0
prob["P_mech"] = 15354206.45251639

# Drivetrain
prob["D_shaft"] = 3.0
prob["D_nose"] = 2.2

# Generator inputs
prob["B_g"] = 1.38963289
prob["B_r"] = 1.279
prob["B_rymax"] = 1.63478983
prob["B_symax"] = 1.63455514
prob["B_tmax"] = 1.88378905
prob["E_p"] = 3300.0 / np.sqrt(3)
prob["L_s"] = 0.01138752
prob["N_c"] = 2
prob["P_Fe0e"] = 1.0
prob["P_Fe0h"] = 4.0
prob["R_s"] = 0.02457052
prob["S"] = 240
prob["S_N"] = -0.002
prob["alpha_p"] = 1.0995574287564276  # 0.5*np.pi*0.7
prob["b"] = 2.0
prob["b_m"] = 0.1288363023
prob["b_r_tau_r"] = 0.45
prob["b_ro"] = 0.004
prob["b_s"] = 0.05121796888
prob["b_s_tau_s"] = 0.45
prob["b_so"] = 0.004
prob["b_t"] = 0.08159225451
prob["c"] = 5.0
prob["cofi"] = 0.85
prob["freq"] = 60.0
prob["h_0"] = 0.005
prob["h_i"] = 0.004
prob["h_m"] = 0.0995385122
prob["h_s"] = 0.37694491492
prob["h_sr"] = 0.04677
prob["h_ss"] = 0.04651
prob["h_sy0"] = 0.0
prob["h_t"] = 0.3859449149
prob["h_w"] = 0.005
prob["h_yr"] = 0.0361759511
prob["h_ys"] = 0.03618114528
prob["k_fes"] = 0.8
prob["k_fillr"] = 0.55
prob["k_fills"] = 0.65
prob["k_s"] = 0.2
prob["len_s"] = 2.23961662
prob["m"] = 3
prob["mu_0"] = 1.2566370614359173e-06  # np.pi*4e-7
prob["mu_r"] = 1.06
prob["p"] = 100
prob["phi"] = 1.5707963267948966  # 90 deg
prob["q1"] = 5
prob["q2"] = 4
prob["rad_ag"] = 0.5*10.25246718
prob["ratio_mw2pp"] = 0.8
prob["resist_Cu"] = 2.52e-8  # 1.8e-8*1.4
prob["sigma"] = 60e3
prob["t_r"] = 0.061
prob["t_s"] = 0.061
prob["tau_p"] = 0.1610453779
prob["tau_s"] = 0.1339360726
prob["u_allow_pcent"] = 8.5
prob["y_allow_pcent"] = 1.0
prob["y_tau_p"] = 0.8  # 12./15.
prob["y_tau_pr"] = 0.8333333  # 10. / 12
prob["z_allow_deg"] = 0.05

# Specific costs
prob["C_Cu"] = 4.786  # Unit cost of Copper $/kg
prob["C_Fe"] = 0.556  # Unit cost of Iron $/kg
prob["C_Fes"] = 0.50139  # specific cost of Structural_mass $/kg
prob["C_PM"] = 95.0

# Material properties
prob["rho_Fe"] = 7700.0  # Steel density Kg/m3
prob["rho_Fes"] = 7850  # structural Steel density Kg/m3
prob["rho_Copper"] = 8900.0  # copper density Kg/m3
prob["rho_PM"] = 7450.0  # typical density Kg/m3 of neodymium magnets (added 2019 09 18) - for pmsg_[disc|arms]
prob["E"] = 2e11
prob["G"] = 79.3e9
# ---

# Run optimization and print results
if opt_flag:
    prob.model.approx_totals()
    prob.run_driver()
else:
    prob.run_model()

prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
# --
fio.save_data('PMSG_OUTER', prob, npz_file=False, mat_file=False, xls_file=True)

