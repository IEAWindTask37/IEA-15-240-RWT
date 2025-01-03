import os
from collections import OrderedDict
from wisdem import run_wisdem
import wisdem.postprocessing.compare_designs as compare_designs
import wisdem.postprocessing.wisdem_get as getter
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
    bladeDF = getter.get_blade_shape(prob)

    # Tabular output: Blade Stiffness
    bladeStiffDF = getter.get_blade_elasticity(prob)

    # Blade internal laminate layer details
    layerDF = []
    l_s = prob.get_val("blade.internal_structure_2d_fem.s")
    lthick = prob.get_val("blade.internal_structure_2d_fem.layer_thickness", 'm')
    lorient = prob.get_val("blade.internal_structure_2d_fem.layer_orientation", 'deg')
    lstart = prob.get_val("blade.internal_structure_2d_fem.layer_start_nd")
    lend = prob.get_val("blade.internal_structure_2d_fem.layer_end_nd")
    nlay = lthick.shape[0]
    layer_cols = ['Span','Thickness [m]','Fiber angle [deg]','Layer Start','Layer End']
    for k in range(nlay):
        ilay = np.c_[l_s, lthick[k,:], lorient[k,:], lstart[k,:], lend[k,:]]
        layerDF.append( pd.DataFrame(data=ilay, columns=layer_cols) )
    
    # Tabular output: Rotor Performance
    perfDF = getter.get_rotor_performance(prob)

    # Nacelle mass properties tabular
    nacDF = getter.get_nacelle_mass(prob)

    # Tabular output: Tower
    towDF = getter.get_tower_table(prob)

    # Frequency plot
    fn_tower = getter.get_tower_freqs(prob)
    f        = np.linspace(0., 0.5, num=1000)[1:]
    omega    = f*(2.*np.pi)

    Omega_rot_min = prob.get_val('control.minOmega', 'rpm')/60.
    Omega_rot_max = prob.get_val('control.maxOmega', 'rpm')/60.

    f_1P = np.r_[0, Omega_rot_min, Omega_rot_min, Omega_rot_max, Omega_rot_max, f[-1]]
    f_3P = np.r_[0, 3.*Omega_rot_min, 3.*Omega_rot_min, 3.*Omega_rot_max, 3.*Omega_rot_max, f[-1]]
    NP_y = np.r_[0., 0., 1., 1., 0., 0.]

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
    overview['Generator rated efficiency [%]'] = prob['rotorse.rp.powercurve.rated_efficiency']
    overview['Blade pre-bend [m]'] = prob['blade.outer_shape_bem.ref_axis'][-1,0]
    overview['Blade mass [t]'] = 1e-3*prob['rotorse.blade_mass']
    overview['Hub mass [t]'] = 1e-3*prob['drivese.hub_mass']
    overview['Generator mass [t]'] = 1e-3*prob['drivese.generator_mass']
    overview['Nacelle mass [t]'] = 1e-3*prob['drivese.nacelle_mass']
    overview['RNA mass [t]'] = 1e-3*prob['drivese.rna_mass']
    overview['Tower mass [t]'] = 1e-3*getter.get_tower_mass(prob)
    zvec = towDF['Height [m]'].to_numpy()
    h_trans = getter.get_transition_height(prob)
    idx = np.where(zvec == h_trans)[0][0]
    overview['Tower base diameter [m]'] = towDF['OD [m]'].iloc[idx]
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
    
