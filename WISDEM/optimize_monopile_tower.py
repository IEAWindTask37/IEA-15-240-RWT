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
h_param = np.array([15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 15., 10.])
#d_param = np.array([10., 10., 10., 9.999994, 9.893298, 9.501227, 9.073816, 8.733734, 8.481259, 8.254697, 8.087231, 7.512527, 6.717548])
#t_param = np.array([0.04922689, 0.04922689, 0.04922689, 0.04581482, 0.04301337, 0.04129422, 0.03939618, 0.03675472, 0.03345327, 0.02984231, 0.02622864, 0.03062863])
d_param = np.array([10., 10., 9.07841266, 8.94626364, 8.2388643, 8.00406996, 7.4747645, 7.16117877, 6.7801444, 6.40652604, 6.14951031, 5.73067083, 5.57747088])
t_param = np.array([0.06470247, 0.05743258, 0.05558765, 0.05632417, 0.05654073, 0.05566434, 0.0556487, 0.05363243, 0.05230072, 0.05033409, 0.04682933, 0.04711726])

z_foundation = -30.0
L_reinforced = 30.0  # [m] buckling length
theta_stress = 0.0
yaw = 0.0
Koutfitting = 1.07
hub_height = 150.0

# --- material props ---
E = 210e9
G = 79.3e9 #80.8e9
rho = 7850.0 #8500.0
sigma_y = 345.0e6 #450.0e6

# --- extra mass ----
m = np.array([1189821.77086906])
mIxx = 3.79640227e+08
mIyy = 2.24477294e+08
mIzz = 1.82971949e+08
mIxy = 0.0
mIxz = 7259625.38388042
mIyz = 0.0
mI = np.array([mIxx, mIyy, mIzz, mIxy, mIxz, mIyz])
mrho = np.array([-1.06144039, 0.0, 0.8916053])
# -----------

# --- wind ---
wind_zref = hub_height
wind_z0 = 0.0
shearExp = 0.11
cd_usr = -1.
# ---------------

# --- wave ---
hmax = 4.52
T = 9.52
cm = 1.0
monopile = True
suction_depth = 45.0
soilG = 140e6
soilnu = 0.4
foundation = -30.0
# ---------------


# two load cases.  TODO: use a case iterator

# # --- loading case 1: max Thrust ---
wind_Uref1 = 20.00138038
Fx1 = 3850904.21819388
Fy1 = -30033.97914658
Fz1 = -11693223.90464621
Mxx1 = 73975385.79607527
Myy1 = 23162187.41183158
Mzz1 = 8170205.14882855
# # ---------------

# --- safety factors ---
gamma_f = 1.35
gamma_m = 1.3
gamma_n = 1.0
gamma_b = 1.1
# ---------------

# --- fatigue ---
z_DEL = None
M_DEL = None
nDEL = 0
gamma_fatigue = 1.35*1.3*1.0
life = 20.0
m_SN = 4
# ---------------


# --- constraints ---
min_d_to_t   = 120.0
max_taper    = 0.2
# ---------------

# # V_max = 80.0  # tip speed
# # D = 126.0
# # .freq1p = V_max / (D/2) / (2*pi)  # convert to Hz

nPoints = len(d_param)
nFull   = 5*(nPoints-1) + 1
wind = 'PowerWind'
nLC = 1

prob = om.Problem()
prob.model = TowerSE(nLC=nLC, nPoints=nPoints, nFull=nFull, wind=wind, topLevelFlag=True, monopile=monopile)
prob.driver = om.pyOptSparseDriver() #om.ScipyOptimizeDriver() # #
prob.driver.options['optimizer'] = 'SNOPT' #'SLSQP' #'CONMIN'

# --- Objective ---
prob.model.add_objective('tower_mass', scaler=1e-6)
# ----------------------

# --- Design Variables ---
prob.model.add_design_var('tower_outer_diameter', lower=3.87, upper=10.0)
prob.model.add_design_var('tower_wall_thickness', lower=4e-3, upper=2e-1)
# ----------------------

# --- Constraints ---
prob.model.add_constraint('height_constraint',    lower=-1e-2,upper=1.e-2)
prob.model.add_constraint('post.stress',          upper=1.0)
prob.model.add_constraint('post.global_buckling', upper=1.0)
prob.model.add_constraint('post.shell_buckling',  upper=1.0)
prob.model.add_constraint('weldability',          upper=0.0)
prob.model.add_constraint('manufacturability',    lower=0.0)
prob.model.add_constraint('slope',    upper=0.0)
prob.model.add_constraint('tower.f1',    lower=0.15)
prob.model.add_constraint('tower.f1',    upper=0.2)
# ----------------------

prob.setup()

if wind=='PowerWind':
    prob['shearExp'] = shearExp

# assign values to params

# --- geometry ----
prob['hub_height'] = hub_height
prob['foundation_height'] = foundation
prob['tower_section_height'] = h_param
prob['tower_outer_diameter'] = d_param
prob['tower_wall_thickness'] = t_param
prob['tower_buckling_length'] = L_reinforced
prob['tower_outfitting_factor'] = Koutfitting
prob['yaw'] = yaw
# prob['monopile'] = monopile
prob['suctionpile_depth'] = suction_depth
prob['soil_G'] = soilG
prob['soil_nu'] = soilnu
# --- material props ---
prob['E'] = E
prob['G'] = G
prob['material_density'] = rho
prob['sigma_y'] = sigma_y

# --- extra mass ----
prob['rna_mass'] = m
prob['rna_I'] = mI
prob['rna_cg'] = mrho
# -----------

# --- wind & wave ---
prob['wind_reference_height'] = wind_zref
prob['wind_z0'] = wind_z0
prob['cd_usr'] = cd_usr
prob['air_density'] = 1.225
prob['air_viscosity'] = 1.7934e-5
prob['water_density'] = 1025.0
prob['water_viscosity'] = 1.3351e-3
prob['wind_beta'] = prob['wave_beta'] = 0.0
prob['significant_wave_height'] = hmax
prob['significant_wave_period'] = T
#prob['waveLoads1.U0'] = prob['waveLoads1.A0'] = prob['waveLoads1.beta0'] = prob['waveLoads2.U0'] = prob['waveLoads2.A0'] = prob['waveLoads2.beta0'] = 0.0
# ---------------

# --- safety factors ---
prob['gamma_f'] = gamma_f
prob['gamma_m'] = gamma_m
prob['gamma_n'] = gamma_n
prob['gamma_b'] = gamma_b
prob['gamma_fatigue'] = gamma_fatigue
# ---------------

prob['DC'] = 80.0
prob['shear'] = True
prob['geom'] = False
prob['tower_force_discretization'] = 5.0
prob['nM'] = 2
prob['Mmethod'] = 1
prob['lump'] = 0
prob['tol'] = 1e-9
prob['shift'] = 0.0


# --- fatigue ---
#prob['tower_z_DEL'] = z_DEL
#prob['tower_M_DEL'] = M_DEL
prob['life'] = life
prob['m_SN'] = m_SN
# ---------------

# --- constraints ---
prob['min_d_to_t'] = min_d_to_t
prob['max_taper'] = max_taper
# ---------------


# # --- loading case 1: max Thrust ---
prob['wind.Uref'] = wind_Uref1

prob['pre.rna_F'] = np.array([Fx1, Fy1, Fz1])
prob['pre.rna_M'] = np.array([Mxx1, Myy1, Mzz1])
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
