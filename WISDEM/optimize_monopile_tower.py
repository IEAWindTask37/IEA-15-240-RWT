import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import openmdao.api as om
from wisdem.towerse.tower import TowerSE
from wisdem.commonse.utilities import assembleI, unassembleI, nodal2sectional

# --- tower setup ------
from wisdem.commonse.environment import PowerWind
from wisdem.commonse.environment import LogWind

# --- geometry ----
h_param = np.array([10., 10., 10., 10., 10., 12.5,  12.5,  12.5,  12.5,  12.5,  12.5,  12.5,  12.5,  12.5, 13.6679-1.58545691])
d_param = np.array([10., 10., 10., 10., 10., 10., 9.8457, 9.47, 9.041, 8.5638, 8.1838, 8.0589, 7.9213, 7.8171, 7.3356, 6.5])
t_param = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.0431, 0.041, 0.0397, 0.0384, 0.037, 0.0348, 0.0313, 0.0279, 0.0248, 0.0299])

# GB: Optimization results using correct RNA forces on Jan 8
d_new = np.array([10., 10., 10., 10., 10., 10., 9.84603509, 9.47076546, 9.0419724, 8.56494627, 8.18565418, 8.06162831, 7.92478627, 7.82113907, 7.33825148, 6.49121862])
t_new = np.array([0.06151846, 0.05619698, 0.05042458, 0.0444563, 0.04175584, 0.03929218, 0.03697146, 0.03528252, 0.03354286, 0.03162747, 0.02898613, 0.02551408, 0.02239292, 0.02025522, 0.02377017])


nPoints = len(d_param)
nFull   = 5*(nPoints-1) + 1

prob = om.Problem()
prob.model = TowerSE(nLC=1, nPoints=nPoints, nFull=nFull, wind='PowerWind', topLevelFlag=True, monopile=True)
prob.driver = om.pyOptSparseDriver() #om.ScipyOptimizeDriver() # #
prob.driver.options['optimizer'] = 'SNOPT' #'SLSQP' #'CONMIN'

# --- Objective ---
prob.model.add_objective('tower_mass', scaler=1e-6)
# ----------------------

# --- Design Variables ---
prob.model.add_design_var('tower_outer_diameter', lower=3.87, upper=9.0)
prob.model.add_design_var('tower_wall_thickness', lower=4e-3, upper=2e-1)
# ----------------------

# --- Constraints ---
prob.model.add_constraint('height_constraint',    lower=-1e-2,upper=1.e-2)
prob.model.add_constraint('post.stress',          upper=1.0)
prob.model.add_constraint('post.global_buckling', upper=1.0)
prob.model.add_constraint('post.shell_buckling',  upper=1.0)
prob.model.add_constraint('weldability',          upper=0.0)
prob.model.add_constraint('manufacturability',    lower=0.0)
prob.model.add_constraint('slope',                upper=1.0)
prob.model.add_constraint('tower.f1',             lower=0.13, upper=0.24)
# ----------------------

prob.setup()


# assign values to params

# --- geometry ----
prob['hub_height'] = prob['wind_reference_height'] = 150.0
prob['foundation_height'] = -30.0
prob['tower_section_height'] = h_param
prob['tower_outer_diameter'] = d_param
prob['tower_wall_thickness'] = t_param
prob['tower_buckling_length'] = 30.0
prob['tower_outfitting_factor'] = 1.07
prob['yaw'] = 0.0
prob['suctionpile_depth'] = 45.0

# --- material props ---
prob['E'] = 210e9
prob['G'] = 79.3e9 #80.8e9
prob['material_density'] = 7850.0 #8500.0
prob['sigma_y'] = 345.0e6 #450.0e6
prob['soil_G'] = 140e6
prob['soil_nu'] = 0.4

# --- extra mass ----
mIxx = 4.10974879e+08
mIyy = 2.73852641e+08
mIzz = 2.10770543e+08
mIxy = 0.0
mIxz = 3.85659547e+07
mIyz = 0.0
prob['rna_mass'] = 1141316.5884164
prob['rna_I'] = np.array([mIxx, mIyy, mIzz, mIxy, mIxz, mIyz])
prob['rna_cg'] = np.array([-7.21526604, 0., 4.47695301])
prob['transition_piece_mass'] = 100e3
prob['transition_piece_height'] = 20.0
prob['tower_add_gravity'] = False # Don't double count
# -----------

# --- wind & wave ---
prob['wind_z0'] = 0.0
prob['air_density'] = 1.225
prob['air_viscosity'] = 1.7934e-5
prob['water_density'] = 1025.0
prob['water_viscosity'] = 1.3351e-3
prob['shearExp'] = 0.11
prob['wind_beta'] = prob['wave_beta'] = 0.0
prob['significant_wave_height'] = 4.52
prob['significant_wave_period'] = 9.52
#prob['waveLoads1.U0'] = prob['waveLoads1.A0'] = prob['waveLoads1.beta0'] = prob['waveLoads2.U0'] = prob['waveLoads2.A0'] = prob['waveLoads2.beta0'] = 0.0
# ---------------

# --- safety factors ---
prob['gamma_f'] = 1.35
prob['gamma_m'] = 1.3
prob['gamma_n'] = 1.0
prob['gamma_b'] = 1.1
prob['gamma_fatigue'] = 1.35*1.3*1.0
# ---------------

# --- frame3dd knobs ---
prob['DC'] = 80.0
prob['shear'] = True
prob['geom'] = True
prob['tower_force_discretization'] = 5.0
prob['nM'] = 2
prob['Mmethod'] = 1
prob['lump'] = 0
prob['tol'] = 1e-9
prob['shift'] = 0.0
# ---------------

# --- fatigue (not used) ---
#prob['tower_z_DEL'] = z_DEL
#prob['tower_M_DEL'] = M_DEL
prob['life'] = 20.0
prob['m_SN'] = 4.0
# ---------------

# --- constraints ---
prob['min_d_to_t'] = 120.0
prob['max_taper']  = 0.2
# ---------------

# # --- loading case 1: max Thrust ---
prob['wind.Uref'] = 20.00138038
prob['pre.rna_F'] = np.array([3796670.10819044,
                              -37378.11861823,
                              -11615856.8796076])
prob['pre.rna_M'] = np.array([75135954.50089163,
                              -61189104.43958881,
                              661646.93272768])
# # ---------------

# # --- run ---
prob.model.approx_totals()
prob.run_driver()

z,_ = nodal2sectional(prob['z_full'])
print('section_height', prob['tower_section_height'])
print('section_diam', prob['tower_outer_diameter'])
print('section_thick', prob['tower_wall_thickness'])
print('zs=', z)
print('ds=', prob['d_full'])
print('ts=', prob['t_full'])
print('mass (kg) =', prob['tower_mass'])
print('cg (m) =', prob['tower_center_of_mass'])
print('weldability =', prob['weldability'])
print('manufacturability =', prob['manufacturability'])
print('\nwind: ', prob['wind.Uref'])
print('f1 (Hz) =', prob['tower.f1'])
print('top_deflection1 (m) =', prob['post.top_deflection'])
print('stress1 =', prob['post.stress'])
print('GL buckling =', prob['post.global_buckling'])
print('Shell buckling =', prob['post.shell_buckling'])


stress1 = np.copy( prob['post.stress'] )
shellBuckle1 = np.copy( prob['post.shell_buckling'] )
globalBuckle1 = np.copy( prob['post.global_buckling'] )
#damage1 = np.copy( prob['post.damage'] )

import matplotlib.pyplot as plt
plt.figure(figsize=(5.0, 3.5))
plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
plt.plot(stress1, z, label='stress 1')
plt.plot(shellBuckle1, z, label='shell buckling 1')
plt.plot(globalBuckle1, z, label='global buckling 1')
#plt.plot(damage1, z, label='damage 1')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
plt.xlabel('utilization')
plt.ylabel('height along tower (m)')

#plt.figure(2)
#plt.plot(prob['d_full']/2.+max(prob['d_full']), z, 'ok')
#plt.plot(prob['d_full']/-2.+max(prob['d_full']), z, 'ok')

#fig = plt.figure(3)
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)

#plt.tight_layout()
plt.show()

print(prob['tower.base_F'])
print(prob['tower.base_M'])
# ------------
