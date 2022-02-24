# -*- coding: utf-8 -*-
"""Functions to create certain types of hawc2 files from a base file

Notes:
    * These functions require the wetb package.
    * The base file should be a 100s, steady-wind simulation with constant shear.
"""
import os
from wetb.hawc2 import HTCFile


def base_to_hs2(orig_htc, new_htc, **kw):
    """Base htc file to a HAWCStab2 file

    Required keyword arguments: cut_in, cut_out, n_wsp,
        gen_min, gen_max, gbr, pitch_min, opt_lambda, rate_pow, gen_eff,
        p1_f, p1_z, p2_f, p2_z, gs, constant_power, oper_dat
    """
    htc = HTCFile(orig_htc)
    # delete useless hawc2 blocks
    del htc.simulation
    del htc.dll
    del htc.output
    del htc.wind.tower_shadow_potential_2  # hawcstab2 can't handle this subblock
    del htc.wind.mann  # don't want hs2 to generate turbulence
    del htc.aerodrag  # can't handle this either
    htc.wind.turb_format = 0
    htc.wind.tower_shadow_method = 0
    # add hawcstab2 block
    hs2 = htc.add_section('hawcstab2')
    # structure
    hs2.add_line('; define structure', [])
    sb = hs2.add_section('ground_fixed_substructure')  # fixed to ground
    sb.main_body = ['tower']
    sb.add_line('main_body', ['towertop'])
    sb.add_line('main_body', ['connector'])
    sb = hs2.add_section('rotating_axissym_substructure')  # rotating with rotor
    sb.main_body = ['shaft']
    sb = hs2.add_section('rotating_threebladed_substructure')  # three-bladed sutbstruct.
    sb.main_body = ['hub1']
    sb.add_line('main_body', ['blade1'])
    hs2.operational_data_filename = [kw['oper_dat']]
    # operational data
    hs2.add_line('; inputs for finding optimal operational data', [])
    sb = hs2.add_section('operational_data')
    sb.add_line('operational_data_file_wind', [1],
                comments='Calculate steady-state values based on wind speeds in opt file')
    sb.add_line('windspeed', [kw['cut_in'], kw['cut_out'], kw['n_wsp']],
                comments='cut-in [m/s], cut-out [m/s], points [-]')
    sb.add_line('genspeed', [kw['gen_min'], kw['gen_max']], comments='[rpm]')
    sb.add_line('gearratio', [kw['gbr']], comments='[-]')
    sb.add_line('minpitch', [kw['pitch_min']], comments='[deg]')
    sb.add_line('opt_lambda', [kw['opt_lambda']], comments='[-]')
    sb.add_line('maxpow', [kw['rate_pow'] / kw['gen_eff']], comments='[kW]')
    sb.add_line('prvs_turbine', [1], comments='[-]')
    sb.add_line('include_torsiondeform', [1], comments='[-]')
    # controller
    hs2.add_line('; define controller couplings', [])
    sb = hs2.add_section('controller')
    sb.add_section('input')
    sb.input.add_line('constraint', ['bearing1', 'shaft_rot'])
    sb.input.add_line('constraint', ['bearing2', 'pitch1', 'collective'])
    sb.add_section('output')
    sb.output.add_line('constraint', ['bearing1', 'shaft_rot', 1, 'only', 2])
    sb.output.add_line('constraint', ['bearing2', 'pitch1', 1, 'only', 1, 'collective'])
    # controller tuning
    hs2.add_line('; inputs for controller tuning', [])
    sb = hs2.add_section('controller_tuning')  # controller tuning block
    sb.partial_load = [kw['p1_f'], kw['p1_z']]
    sb.partial_load.comments = 'fn [hz], zeta [-]'
    sb.full_load = [kw['p2_f'], kw['p2_z']]
    sb.full_load.comments = 'fn [hz], zeta [-]'
    sb.gain_scheduling = [kw['gs']]
    sb.gain_scheduling.comments = '1 linear, 2 quadratic'
    sb.constant_power = [kw['constant_power']]
    sb.constant_power.comments = '0 constant torque, 1 constant power at full load'
    # HAWC2S commands
    hs2.add_line('; HAWC2S commands (uncomment as needed)', [])
    line = hs2.add_line(';compute_optimal_pitch_angle', ['use_operational_data'])
    line = hs2.add_line(';compute_steady_states', ['bladedeform', 'tipcorrect',
                                                  'induction', 'nogradients'])
    line.comments = 'compute steady states using hawcstab2 (need for other commands)'
    line = hs2.add_line(';save_power', [])
    line.comments = 'save steady-state values to input_htc_file.pwr'
    line = hs2.add_line(';compute_structural_modal_analysis', ['nobladeonly', 12])
    line.comments = 'struct modal analysis saved to <htc>_struc.cmb'
    line = hs2.add_line(';compute_stability_analysis', ['windturbine', 12])
    line.comments = 'aeroelastic modal analysis saved to <htc>.cmb'
    line = hs2.add_line(';compute_controller_input', [])
    line.comments = 'tune controller parameters'
    # save the new file
    htc.save(new_htc)


def base_to_step(orig_htc, new_htc, **kw):
    """Base htc file to a step wind (no tower shadow or shear)

    Required keyword arguments: cut_in, cut_out, dt, tstart
    """
    fname = os.path.basename(orig_htc).replace('.htc', '')
    htc = HTCFile(orig_htc)
    wsp = range(kw['cut_in'], kw['cut_out'] + 1)
    # set simulation time and file names
    time_stop = (len(wsp) - 1) * (kw['dt'] + 1) + kw['tstart'] + kw['dt']
    htc.simulation.time_stop = time_stop
    htc.simulation.logfile = f'./log/{fname}_step.log'
    htc.output.filename = f'./res/{fname}_step'
    # add the step wind
    del htc.wind.mann  # remove mann block
    htc.wind.wsp = wsp[0]  # set first wsp
    htc.wind.tint = 0  # no turbulence intensity
    htc.wind.shear_format = [3, 0.0]  # no shear
    htc.wind.turb_format = 0
    htc.wind.tower_shadow_method = 0  # no tower shadow
    htc.wind.add_line(';', ['step-wind for testing controller tuning'])
    for i, u in enumerate(wsp[1:]):
        t0 = (i + 1) * kw['dt'] + i + kw['tstart']
        tend = t0 + 1
        line = htc.wind.add_line('wind_ramp_abs', [t0, tend, 0, wsp[i + 1] - wsp[i]])
        line.comments = f'wsp. after the step:  {u:.1f}'
    # update the simulation time
    htc.output.time = [kw['tstart'], time_stop]
    # save the new file
    htc.save(new_htc)


def base_to_turb(orig_htc, new_htc, tower_shadow=3, **kw):
    """Base htc file to a 700-s Mann turbulence file w/power law

    Required keyword arguments: wsp, tint, tb_wid, tb_ht
    """
    nx, ny, nz = 8192, 32, 32
    fname = os.path.basename(orig_htc).replace('.htc', '')
    htc = HTCFile(orig_htc)
    U, tint = kw['wsp'], kw['tint']
    wid, ht = kw['tb_wid'], kw['tb_ht']
    # set simulation time and file names
    htc.simulation.time_stop = 700
    htc.simulation.logfile = f'./log/{fname}_turb.log'
    htc.output.filename = f'./res/{fname}_turb'
    # add the step wind
    htc.wind.wsp = U  # mean wind speed
    htc.wind.tint = tint  # turbulence intensity
    htc.wind.shear_format = [3, 0.2]  # no shear
    htc.wind.turb_format = 1  # mann
    htc.wind.tower_shadow_method = tower_shadow
    mann = htc.wind.add_section('mann')
    mann.add_line('create_turb_parameters', [29.4, 1.0, 3.9, 1001, 0])
    mann.add_line('filename_u', [f'./turb/{fname}_turb_u.bin'])
    mann.add_line('filename_v', [f'./turb/{fname}_turb_v.bin'])
    mann.add_line('filename_w', [f'./turb/{fname}_turb_w.bin'])
    mann.add_line('box_dim_u', [nx, U * 600 / nx])  # repeat every 600 seconds
    mann.add_line('box_dim_v', [ny, wid / (ny - 1)])
    mann.add_line('box_dim_w', [nz, ht / (nz - 1)])
    # update the simulation time
    htc.output.time = [100, 700]
    # save the new file
    htc.save(new_htc)