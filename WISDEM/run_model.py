import os
from collections import OrderedDict
from wisdem import run_wisdem
import wisdem.postprocessing.compare_designs as compare_designs
import wisdem.postprocessing.wisdem_get as getter
import wisdem.commonse.utilities as util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generateTables import RWT_Tabular

# File management
thisdir = os.path.dirname(os.path.realpath(__file__))
ontology_dir = os.path.join(os.path.dirname(thisdir), "WT_Ontology")
fname_modeling_options = os.path.join(thisdir, "modeling_options.yaml")
fname_analysis_options = os.path.join(thisdir, "analysis_options.yaml")
folder_output = os.path.join(thisdir, "outputs")

def run_15mw(fname_wt_input):
    float_flag = fname_wt_input.find('Volturn') >= 0
    
    # Run WISDEM
    prob, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

    # Produce standard plots
    compare_designs.run([prob], ['IEA Wind 15-MW'], modeling_options, analysis_options)

    # Tabular output: Blade Shape
    blade_shape  = np.c_[prob.get_val('blade.outer_shape_bem.s'),
                         prob.get_val('blade.outer_shape_bem.ref_axis','m')[:,2],
                         prob.get_val('blade.outer_shape_bem.chord','m'),
                         prob.get_val('blade.outer_shape_bem.twist', 'deg'),
                         prob.get_val('blade.interp_airfoils.r_thick_interp')*100,
                         prob.get_val('blade.outer_shape_bem.pitch_axis')*100,
                         prob.get_val('blade.outer_shape_bem.ref_axis','m')[:,0],
                         prob.get_val('blade.outer_shape_bem.ref_axis','m')[:,1],
                         ]
    blade_shape_col = ['Blade Span','Rotor Coordinate [m]',
                       'Chord [m]', 'Twist [deg]',
                       'Relative Thickness [%]', 'Pitch Axis Chord Location [%]',
                       'Prebend [m]', 'Sweep [m]']
    bladeDF = pd.DataFrame(data=blade_shape, columns=blade_shape_col)

    # Tabular output: Blade Stiffness
    blade_stiff  = np.c_[prob.get_val('rotorse.r','m'),
                         prob.get_val('rotorse.A','m**2'),
                         prob.get_val('rotorse.EA','N'),
                         prob.get_val('rotorse.EIxx','N*m**2'),
                         prob.get_val('rotorse.EIyy','N*m**2'),
                         prob.get_val('rotorse.EIxy','N*m**2'),
                         prob.get_val('rotorse.GJ','N*m**2'),
                         prob.get_val('rotorse.rhoA','kg/m'),
                         prob.get_val('rotorse.rhoJ','kg*m'),
                         prob.get_val('rotorse.x_ec','mm'),
                         prob.get_val('rotorse.y_ec','mm'),
                         prob.get_val('rotorse.re.x_tc','mm'),
                         prob.get_val('rotorse.re.y_tc','mm'),
                         prob.get_val('rotorse.re.x_sc','mm'),
                         prob.get_val('rotorse.re.y_sc','mm'),
                         prob.get_val('rotorse.re.x_cg','mm'),
                         prob.get_val('rotorse.re.y_cg','mm'),
                         prob.get_val('rotorse.re.precomp.flap_iner','kg/m'),
                         prob.get_val('rotorse.re.precomp.edge_iner','kg/m')]
    blade_stiff_col = ['Blade Span [m]',
                       'Cross-sectional area [m^2]',
                       'Axial stiffness [N]',
                       'Edgewise stiffness [Nm^2]',
                       'Flapwise stiffness [Nm^2]',
                       'Flap-edge coupled stiffness [Nm^2]',
                       'Torsional stiffness [Nm^2]',
                       'Mass density [kg/m]',
                       'Polar moment of inertia density [kg*m]',
                       'X-distance to elastic center [mm]',
                       'Y-distance to elastic center [mm]',
                       'X-distance to tension center [mm]',
                       'Y-distance to tension center [mm]',
                       'X-distance to shear center [mm]',
                       'Y-distance to shear center [mm]',
                       'X-distance to mass center [mm]',
                       'Y-distance to mass center [mm]',
                       'Section flap inertia [kg/m]',
                       'Section edge inertia [kg/m]',
                       ]
    bladeStiffDF = pd.DataFrame(data=blade_stiff, columns=blade_stiff_col)

    # Blade internal laminate layer details
    layerDF = []
    l_s = prob.get_val("blade.internal_structure_2d_fem.s")
    lthick = prob.get_val("blade.internal_structure_2d_fem.layer_thickness", 'm')
    lrot = prob.get_val("blade.internal_structure_2d_fem.layer_rotation", 'deg')
    lstart = prob.get_val("blade.internal_structure_2d_fem.layer_start_nd")
    lend = prob.get_val("blade.internal_structure_2d_fem.layer_end_nd")
    nlay = lthick.shape[0]
    layer_cols = ['Span','Thickness [m]','Fiber angle [deg]','Layer Start','Layer End']
    for k in range(nlay):
        ilay = np.c_[l_s, lthick[k,:], lrot[k,:], lstart[k,:], lend[k,:]]
        layerDF.append( pd.DataFrame(data=ilay, columns=layer_cols) )
    
    # Tabular output: Rotor Performance
    rotor_perf = np.c_[prob.get_val("rotorse.rp.powercurve.V",'m/s'),
                       prob.get_val("rotorse.rp.powercurve.pitch",'deg'),
                       prob.get_val("rotorse.rp.powercurve.P",'MW'),
                       prob.get_val("rotorse.rp.powercurve.Cp"),
                       prob.get_val("rotorse.rp.powercurve.Cp_aero"),
                       prob.get_val("rotorse.rp.powercurve.Omega",'rpm'),
                       prob.get_val("rotorse.rp.powercurve.Omega",'rad/s')*0.5*prob["configuration.rotor_diameter_user"],
                       prob.get_val("rotorse.rp.powercurve.T",'MN'),
                       prob.get_val("rotorse.rp.powercurve.Ct_aero"),
                       prob.get_val("rotorse.rp.powercurve.Q",'MN*m'),
                       prob.get_val("rotorse.rp.powercurve.Cq_aero"),
                       prob.get_val("rotorse.rp.powercurve.M",'MN*m'),
                       prob.get_val("rotorse.rp.powercurve.Cm_aero"),
                       ]
    rotor_perf_col = ['Wind [m/s]','Pitch [deg]',
                      'Power [MW]','Power Coefficient [-]','Aero Power Coefficient [-]',
                      'Rotor Speed [rpm]','Tip Speed [m/s]',
                      'Thrust [MN]','Thrust Coefficient [-]',
                      'Torque [MNm]','Torque Coefficient [-]',
                      'Blade Moment [MNm]','Blade Moment Coefficient [-]',
                      ]
    perfDF = pd.DataFrame(data=rotor_perf, columns=rotor_perf_col)

    # Nacelle mass properties tabular
    # Columns are ['Mass', 'CoM_x', 'CoM_y', 'CoM_z',
    #              'MoI_cm_xx', 'MoI_cm_yy', 'MoI_cm_zz', 'MoI_cm_xy', 'MoI_cm_xz', 'MoI_cm_yz',
    #              'MoI_TT_xx', 'MoI_TT_yy', 'MoI_TT_zz', 'MoI_TT_xy', 'MoI_TT_xz', 'MoI_TT_yz']
    nacDF = prob.model.wt.drivese.nac._mass_table
    hub_cm = float(prob["drivese.hub_system_cm"])
    L_drive = float(prob["drivese.L_drive"])
    tilt = float(prob.get_val('nacelle.uptilt', 'rad'))
    shaft0 = prob["drivese.shaft_start"]
    Cup = -1.0
    hub_cm = R = shaft0 + (L_drive + hub_cm) * np.array([Cup * np.cos(tilt), 0.0, np.sin(tilt)])
    hub_mass = prob['drivese.hub_system_mass']
    hub_I = prob["drivese.hub_system_I"]
    hub_I_TT = util.rotateI(hub_I, -Cup * tilt, axis="y")
    hub_I_TT = util.unassembleI( util.assembleI(hub_I_TT) +
                                 hub_mass * (np.dot(R, R) * np.eye(3) - np.outer(R, R)) )
    blades_mass = prob['drivese.blades_mass']
    blades_I = prob["drivese.blades_I"]
    blades_I_TT = util.rotateI(blades_I, -Cup * tilt, axis="y")
    blades_I_TT = util.unassembleI( util.assembleI(blades_I_TT) +
                                    blades_mass * (np.dot(R, R) * np.eye(3) - np.outer(R, R)) )
    rna_mass = prob['drivese.rna_mass']
    rna_cm = R = prob['drivese.rna_cm']
    rna_I_TT = prob['drivese.rna_I_TT']
    rna_I = util.unassembleI( util.assembleI(rna_I_TT) +
                                    rna_mass * (np.dot(R, R) * np.eye(3) + np.outer(R, R)) )
    nacDF.loc['Blades'] = np.r_[blades_mass, hub_cm, blades_I, blades_I_TT].tolist()
    nacDF.loc['Hub_System'] = np.r_[hub_mass, hub_cm, hub_I, hub_I_TT].tolist()
    nacDF.loc['RNA'] = np.r_[rna_mass, rna_cm, rna_I, rna_I_TT].tolist()

    # Tabular output: Tower
    water_depth = prob['env.water_depth']
    h_trans = getter.get_transition_height(prob)
    htow = getter.get_zpts(prob) #np.cumsum(np.r_[0.0, prob['towerse.tower_section_height']]) + prob['towerse.z_start']
    t = getter.get_tower_thickness(prob)
    towdata = np.c_[htow,
                    getter.get_tower_diameter(prob),
                    np.r_[t[0], t]]
    rowadd = []
    for k in range(towdata.shape[0]):
        if k==0: continue
        if k+1 < towdata.shape[0]:
            rowadd.append([towdata[k,0]+1e-3, towdata[k,1], towdata[k+1,2]])
    towdata = np.vstack((towdata, rowadd))
    towdata[:,-1] *= 1e3
    towdata = np.round( towdata[towdata[:,0].argsort(),], 3)
    colstr = ['Height [m]','OD [m]', 'Thickness [mm]']
    towDF = pd.DataFrame(data=towdata, columns=colstr)
    mycomments = ['']*towdata.shape[0]
    if not float_flag:
        #breakpoint()
        mycomments[0] = 'Monopile start'
        mycomments[np.where(towdata[:,0] == -water_depth)[0][0]] = 'Mud line'
        mycomments[np.where(towdata[:,0] == 0.0)[0][0]] = 'Water line'
    mycomments[np.where(towdata[:,0] == h_trans)[0][0]] = 'Tower start'
    mycomments[-1] = 'Tower top'
    towDF['Location'] = mycomments
    towDF = towDF[['Location']+colstr]
    A = 0.25*np.pi*(towDF['OD [m]']**2 - (towDF['OD [m]']-2*1e-3*towDF['Thickness [mm]'])**2)
    I = (1/64.)*np.pi*(towDF['OD [m]']**4 - (towDF['OD [m]']-2*1e-3*towDF['Thickness [mm]'])**4)
    towDF['Mass Density [kg/m]'] = getter.get_tower_rho(prob)[0] * A
    towDF['Fore-aft inertia [kg.m]'] = towDF['Side-side inertia [kg.m]'] = towDF['Mass Density [kg/m]'] * I/A
    towDF['Fore-aft stiffness [N.m^2]'] = towDF['Side-side stiffness [N.m^2]'] = getter.get_tower_E(prob)[0] * I
    towDF['Torsional stiffness [N.m^2]'] = getter.get_tower_G(prob)[0] * 2*I
    towDF['Axial stiffness [N]'] = getter.get_tower_E(prob)[0] * A
    #with open('tow.tbl','w') as f:
    #    towDF.to_latex(f, index=False)

    # Frequency plot
    fn_tower = getter.get_tower_freqs(prob)
    f        = np.linspace(0., 0.5, num=1000)[1:]
    omega    = f*(2.*np.pi)

    Omega_rot_min = prob.get_val('control.minOmega', 'rpm')/60.
    Omega_rot_max = prob.get_val('control.maxOmega', 'rpm')/60.

    f_1P = [0, Omega_rot_min, Omega_rot_min, Omega_rot_max, Omega_rot_max, f[-1]]
    f_3P = [0, 3.*Omega_rot_min, 3.*Omega_rot_min, 3.*Omega_rot_max, 3.*Omega_rot_max, f[-1]]
    NP_y = [0., 0., 1., 1., 0., 0.]

    # f_1P_marg_low  = Omega_rot_min/1.1
    # f_1P_marg_high = Omega_rot_min*1.1
    # f_3P_marg_low  = Omega_rot_min*3./1.1
    # f_3P_marg_high = Omega_rot_min*3.*1.1

    f_tower = [fn_tower[0], fn_tower[0]]
    y_tower = [0., 1.]

    # Kaimal
    Sigma1 = 42
    V_hub  = 10.
    L_k = 8.1*Sigma1
    Sk = np.zeros_like(f)
    for i, fi in enumerate(f):
        Sk[i] = 4*fi*L_k/V_hub/(1+6*fi*L_k/V_hub)**(5./3.)/fi
    Sk /= max(Sk)

    # Pierson-Moskowitz
    F = 500000
    g = 9.81
    U10 = 10.

    U19     = U10*1.17
    alpha   = 8.1e-3
    Beta    = 0.74
    omega_0 = g/U19
    omega_p = 0.877*g/U19

    S = np.zeros_like(f)
    for i, omega_i in enumerate(omega):
        S[i] = alpha*g**2./omega_i**5.*np.exp(-Beta*(omega_0/omega_i)**4.)
    S /= max(S)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6., 2.5))
    fig.subplots_adjust(bottom=0.36, top=0.975,  left=0.12, right=0.975)#, right=0.8, hspace=0.3, wspace=0.3)

    ax.fill(f_1P, NP_y, color=[0.5,0.5,0.5], label='1P')
    ax.fill(f_3P, NP_y, color=[0.75,0.75,0.75], label='3P')
    ax.plot(f, Sk, color='r', label='Wind,\nKaimal Spect.', linewidth=1.5)
    ax.plot(f, S,  color='b', label="Waves,\nJONSWAP Spect.", linewidth=1.5)
    ax.plot(f_tower, y_tower, color='k', label="Tower,\n1st Nat. Freq.", linewidth=1.5)
    ax.grid(color=[0.8,0.8,0.8], linestyle='--')
    ax.set_ylabel('Normalized PSD')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_xlim((0.,max(f)))

    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=5, prop={'size':9})
    fig_name = 'tower_fn'
    fig.savefig(folder_output + os.sep + fig_name+'.pdf', pad_inches=0.1, bbox_inches='tight')
    fig.savefig(folder_output + os.sep + fig_name+'.png', pad_inches=0.1, bbox_inches='tight')

    # Overview dictionary
    overview = OrderedDict()

    overview['Power rating [MW]'] = prob.get_val('configuration.rated_power','MW')
    overview['Turbine class'] = 'IEC Class '+ prob['configuration.ws_class']+prob['configuration.turb_class']
    overview['Rotor diameter [m]'] = prob["configuration.rotor_diameter_user"]
    overview['Specific rating [W/m^2]'] = prob.get_val('configuration.rated_power','W')/np.pi/(0.5*prob["configuration.rotor_diameter_user"])**2.0
    overview['Rotor orientation'] = 'Upwind' if prob['configuration.upwind'] else 'Downwind'
    overview['Number of blades'] = prob['configuration.n_blades']
    overview['Control'] = 'Variable speed, Collective pitch'
    overview['Cut-in wind speed [m/s]'] = prob['control.V_in']
    overview['Rated wind speed [m/s]'] = prob["rotorse.rp.powercurve.rated_V"]
    overview['Cut-out wind speed [m/s]'] = prob['control.V_out']
    overview['Airfoil series'] = 'FFA-W3'
    overview['Hub height [m]'] = prob['configuration.hub_height_user']
    overview['Hub diameter [m]'] = prob['hub.diameter']
    overview['Hub Overhang [m]'] = -prob['nacelle.overhang']
    overview['Drive train'] = 'Low speed, Direct drive'
    overview['Design tip speed ratio'] = prob['control.rated_TSR']
    overview['Minimum rotor speed [rpm]'] = prob.get_val('control.minOmega','rpm')
    overview['Maximum rotor speed [rpm]'] = prob.get_val('control.maxOmega','rpm')
    overview['Maximum tip speed [m/s]'] = prob['control.max_TS']
    overview['Shaft tilt angle [deg]'] = prob.get_val('nacelle.uptilt','deg')
    overview['Rotor cone angle [deg]'] = prob.get_val('hub.cone','deg')
    overview['Tower top to hub flange height [m]'] = prob['nacelle.distance_tt_hub']
    overview['Generator rated efficiency [%]'] = prob["drivese.generator_efficiency"][-1]
    overview['Blade pre-bend [m]'] = prob['blade.outer_shape_bem.ref_axis'][-1,0]
    overview['Blade mass [t]'] = 1e-3*prob['rotorse.re.precomp.blade_mass']
    overview['Hub mass [t]'] = 1e-3*prob['drivese.hub_mass']
    overview['Generator mass [t]'] = 1e-3*prob['drivese.generator_mass']
    overview['Nacelle mass [t]'] = 1e-3*prob['drivese.nacelle_mass']
    overview['RNA mass [t]'] = 1e-3*prob['drivese.rna_mass']
    overview['Tower mass [t]'] = 1e-3*getter.get_tower_mass(prob)
    overview['Tower base diameter [m]'] = towdata[np.where(towdata[:,0] == h_trans)[0][0],1]
    overview['Transition piece height [m]'] = h_trans
    if not float_flag:
        overview['Monopile embedment depth [m]'] = prob['fixedse.suctionpile_depth']
        overview['Monopile base diameter [m]'] = prob['fixedse.monopile_outer_diameter'][0]
        overview['Monopile mass [t]'] = 1e-3*prob['fixedse.monopile_mass']
    else:
        overview['Volturn-S hull mass [t]'] = 1e-3*prob['floatingse.platform_hull_mass']
        overview['Volturn-S fixed ballast mass [t]'] = 1e-3*prob['floatingse.platform_ballast_mass'][0]
        overview['Volturn-S fluid ballast mass [t]'] = 1e-3*prob['floatingse.variable_ballast_mass'][0]
        overview['Volturn-S displacement [m^3]'] = prob['floatingse.platform_displacement']
        overview['Volturn-S freeboard [m]'] = 15.0
        overview['Volturn-S draft [m]'] = 20.0

    # Write all tabular data to xlsx
    myobj = RWT_Tabular(fname_wt_input, towDF=towDF, rotDF=perfDF,
                        nacDF=nacDF, layerDF=layerDF, overview=overview)
    myobj.write_all()


if __name__ == '__main__':
    run_15mw( os.path.join(ontology_dir, "IEA-15-240-RWT.yaml") )
    run_15mw( os.path.join(ontology_dir, "IEA-15-240-RWT_VolturnUS-S.yaml") )
    
