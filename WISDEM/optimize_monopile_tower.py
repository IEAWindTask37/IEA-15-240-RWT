import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import openmdao.api as om
from wisdem.towerse.tower import TowerSE
from wisdem.commonse.utilities import assembleI, unassembleI, nodal2sectional

# --- tower setup ------
from wisdem.commonse.environment import PowerWind
from wisdem.commonse.environment import LogWind

h_param = np.array([5., 5., 5., 5., 5., 5., 5., 5., 5., 13.,  13.,  13.,  13.,  13.,  13.,  13.,  13.,  13., 12.58244309])
d_param = np.array([10., 9.8030784, 9.79874123, 9.8030784, 9.79874123, 9.8030784, 9.68185956, 9.44717576, 9.22513042, 9., 9., 9., 9., 9., 9., 9., 9., 9., 7.28693747, 6.5])
t_param = np.array([0.05664958, 0.05562889, 0.05373922, 0.05177922, 0.04983506, 0.04793076, 0.04676483, 0.04677619, 0.04678032, 0.075, 0.06877404, 0.06040807, 0.05194785, 0.04336537, 0.0347967, 0.02639276, 0.02241118, 0.02142537, 0.02633759])

itow = 9




def set_common_params(prob):
    # --- geometry ----
    prob['hub_height'] = prob['wind_reference_height'] = 150.0
    prob['tower_buckling_length'] = 15.0
    prob['tower_outfitting_factor'] = 1.07
    prob['yaw'] = 0.0

    # --- material props ---
    prob['E'] = 210e9
    prob['G'] = 79.3e9 #80.8e9
    prob['material_density'] = 7850.0 #8500.0
    prob['sigma_y'] = 345.0e6 #450.0e6

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
    prob['tower_add_gravity'] = True
    # -----------

    # --- wind & wave ---
    prob['wind_z0'] = 0.0
    prob['air_density'] = 1.225
    prob['air_viscosity'] = 1.7934e-5
    prob['shearExp'] = 0.11
    prob['wind_beta'] = prob['wave_beta'] = 0.0
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
    prob['life'] = 25.0
    prob['m_SN'] = 4.0
    # ---------------

    # --- constraints ---
    prob['min_d_to_t'] = 120.0
    prob['max_taper']  = 0.2
    # ---------------

    # # --- loading case 1: max Thrust ---
    prob['wind.Uref'] = 20.00138038
    prob['pre.rna_F'] = np.array([3712210.30517985,
                                  -25397.27352212,
                                  -419019.0213100216])
    prob['pre.rna_M'] = np.array([71685475.4824221,
                                  17184504.66608971,
                                  1765885.10839113])
    # # ---------------
    return prob


def postprocess(prob):
    z,_ = nodal2sectional(prob['z_full'])
    print('section_height [m]', prob['tower_section_height'])
    print('section_diam [m]', prob['tower_outer_diameter'])
    print('section_thick [m]', prob['tower_wall_thickness'])
    print('pile depth [m]', prob['suctionpile_depth'])
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
    print(prob['tower.base_F'])
    print(prob['tower.base_M'])

    '''
    stress1 = np.copy( prob['post.stress'] )
    shellBuckle1 = np.copy( prob['post.shell_buckling'] )
    globalBuckle1 = np.copy( prob['post.global_buckling'] )

    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(stress1, z, label='stress 1')
    ax1.plot(shellBuckle1, z, label='shell buckling 1')
    ax1.plot(globalBuckle1, z, label='global buckling 1')
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
    ax1.set_xlabel('utilization')
    ax1.set_ylabel('height along tower (m)')

    ax2.plot(prob['d_full']/2.+max(prob['d_full']), prob['z_full'], 'ok')
    ax2.plot(prob['d_full']/-2.+max(prob['d_full']), prob['z_full'], 'ok')
    plt.show()
    '''


    
def design_floating_tower():

    # Optimize a fixed bottom tower with the frequency range such that when placed on a floating platform, the frequencies shift to not align with 1P/3P bounds

    # Set common and then customized parameters
    nPoints = len(d_param[itow:])
    nFull   = 5*(nPoints-1) + 1

    prob = om.Problem()
    prob.model = TowerSE(nLC=1, nPoints=nPoints, nFull=nFull, wind='PowerWind', topLevelFlag=True, monopile=False)
    prob.driver = om.pyOptSparseDriver() #om.ScipyOptimizeDriver() # 
    prob.driver.options['optimizer'] = 'SNOPT' #'SLSQP' #'CONMIN'

    # --- Objective ---
    prob.model.add_objective('tower_mass', scaler=1e-6)
    # ----------------------

    # --- Design Variables ---
    prob.model.add_design_var('tower_outer_diameter', lower=3.87, upper=9.0, indices=[m for m in range(nPoints-1)])
    prob.model.add_design_var('tower_wall_thickness', lower=4e-3, upper=2e-1)
    # ----------------------

    # --- Constraints ---
    #prob.model.add_constraint('height_constraint',    lower=-1e-2,upper=1.e-2)
    prob.model.add_constraint('post.stress',          upper=1.0)
    prob.model.add_constraint('post.global_buckling', upper=1.0)
    prob.model.add_constraint('post.shell_buckling',  upper=1.0)
    prob.model.add_constraint('weldability',          upper=0.0)
    prob.model.add_constraint('manufacturability',    lower=0.0)
    prob.model.add_constraint('slope',                upper=1.0)
    prob.model.add_constraint('tower.f1',             lower=0.28)#lower=0.09, upper=0.15)
    # ----------------------

    prob.setup()
    
    prob = set_common_params(prob)
    prob['foundation_height'] = 0.0
    prob['tower_section_height'] = h_param[itow:]
    prob['tower_outer_diameter'] = d_param[itow:]
    prob['tower_wall_thickness'] = t_param[itow:]
    prob['suctionpile_depth'] = 0.0
    prob['transition_piece_mass'] = 1e-3
    prob['transition_piece_height'] = 0.0
    prob['soil_G'] = 1e30
    prob['soil_nu'] = 0.0
    # Floating will have higher loading
    coeff = 1.25
    prob['pre.rna_F'][:2] *= coeff
    prob['pre.rna_M'] *= coeff

    # Run optimization
    prob.model.approx_totals()
    prob.run_driver()
    print('-----FLOATING TOWER RESULTS---------')
    postprocess(prob)
    
    return prob



def design_monopile_tower(floating_tower=True):

    nPoints = len(d_param)
    nFull   = 5*(nPoints-1) + 1

    prob = om.Problem()
    prob.model = TowerSE(nLC=1, nPoints=nPoints, nFull=nFull, wind='PowerWind', topLevelFlag=True, monopile=True)
    prob.driver = om.pyOptSparseDriver() #om.ScipyOptimizeDriver() # 
    prob.driver.options['optimizer'] = 'SNOPT' #'SLSQP' #'CONMIN'

    # --- Objective ---
    prob.model.add_objective('tower_mass', scaler=1e-6)
    # ----------------------

    # --- Design Variables ---
    if floating_tower:
        prob.model.add_design_var('tower_outer_diameter', lower=3.87, upper=10.0, indices=[m for m in range(itow)])
        prob.model.add_design_var('tower_wall_thickness', lower=4e-3, upper=2e-1, indices=[m for m in range(itow)])
    else:
        prob.model.add_design_var('tower_outer_diameter', lower=3.87, upper=10.0, indices=[m for m in range(nPoints-1)])
        prob.model.add_design_var('tower_wall_thickness', lower=4e-3, upper=2e-1)
    #prob.model.add_design_var('suctionpile_depth', lower=10., upper=70.)
    # ----------------------

    # --- Constraints ---
    #prob.model.add_constraint('height_constraint',    lower=-1e-2,upper=1.e-2)
    prob.model.add_constraint('post.stress',          upper=1.0)
    prob.model.add_constraint('post.global_buckling', upper=1.0)
    prob.model.add_constraint('post.shell_buckling',  upper=1.0)
    prob.model.add_constraint('weldability',          upper=0.0)
    prob.model.add_constraint('manufacturability',    lower=0.0)
    prob.model.add_constraint('slope',                upper=1.0)
    prob.model.add_constraint('tower.f1',             lower=0.13, upper=0.24)
    # ----------------------

    prob.setup()
    
    # Set common and then customized parameters
    prob = set_common_params(prob)
    prob['foundation_height'] = -30.0
    prob['tower_section_height'] = h_param
    prob['tower_outer_diameter'] = d_param
    prob['tower_wall_thickness'] = t_param
    prob['suctionpile_depth'] = 45.0
    prob['transition_piece_mass'] = 100e3
    prob['transition_piece_height'] = 15.0
    prob['soil_G'] = 140e6
    prob['soil_nu'] = 0.4
    prob['air_viscosity'] = 1.7934e-5
    prob['water_density'] = 1025.0
    prob['water_viscosity'] = 1.3351e-3
    prob['significant_wave_height'] = 4.52
    prob['significant_wave_period'] = 9.52

    # Keep tower suitable for floating as static design
    if floating_tower:
        prob0 = design_floating_tower()
        prob['tower_outer_diameter'][itow:] = prob0['tower_outer_diameter']
        prob['tower_wall_thickness'][itow:] = prob0['tower_wall_thickness']
    else:
        # Make the optimizer work a little less hard by using a better starting point
        prob['tower_outer_diameter'] = np.array([10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 9.92647687, 9.44319282, 8.83283769, 8.15148167, 7.38976138, 6.90908962, 6.74803581, 6.57231775, 6.5])
        prob['tower_wall_thickness'] = np.array([0.05534138, 0.05344902, 0.05150928, 0.04952705, 0.04751736, 0.04551709, 0.0435267, 0.04224176, 0.04105759, 0.0394965, 0.03645589, 0.03377851, 0.03219233, 0.03070819, 0.02910109, 0.02721289, 0.02400931, 0.0208264, 0.02399756])

        



    # Run optimization
    prob.model.approx_totals()
    prob.run_driver()
    print('-----MONOPILE TOWER RESULTS---------')
    postprocess(prob)

    return prob


if __name__ == '__main__':
    #prob_float = design_monopile_tower(floating_tower=True)
    # Determine the penalty for using a single tower as opposed to two different ones
    prob_mono  = design_monopile_tower(floating_tower=False)
    #print(prob_mono['tower_mass'] - prob_float['tower_mass'])
