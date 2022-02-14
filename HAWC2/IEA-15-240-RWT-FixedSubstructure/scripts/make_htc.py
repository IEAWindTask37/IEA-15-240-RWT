# -*- coding: utf-8 -*-
"""Create step, turb, and/or hawcstab2 htc files from a base file.

Step and turb htc files will be created in htc directory.
HAWCStab2 file will be created directly in model directory.
Default files (base and hawcstab2) are for FPM model with torsion.

Requirements: Python 3.6+ and wetb package.
"""
from _htc_conversion_fxns import base_to_hs2, base_to_step, base_to_turb


make_step = True  # make a step-wind file?
make_turb = True  # make a turbulence file?
make_hs2 = True  # make a hawcstab2 file?
base_htc = '../htc/IEA_15MW_RWT_FixedSubstructure.htc'  # relative path to base htc file

kw = dict(cut_in=3, cut_out=25, dt=39, tstart=220,  # step parameters
          wsp=12, tint=0.17, tb_wid=252, tb_ht=252,  # turbulence parameters
          n_wsp=45, gen_min=5.000011692174984, gen_max=7.559987120819503,  # hawcstab2 parameters
          gbr=1, pitch_min=0, opt_lambda=9, rate_pow=15e3, gen_eff=0.9655,  # hawcstab2 parameters
          p1_f=0.05, p1_z=0.7, p2_f=0.03, p2_z=0.7, gs=2, constant_power=0,  # hawcstab2 parameters
          oper_dat='./data/IEA_15MW_RWT_FixedSubstructure.opt')  # hawcstab2 parameters


if __name__ == '__main__':

    if make_step:
        step_htc = base_htc.replace('.htc', '_step.htc')
        print(f' Writing step-wind file to {step_htc}...')
        base_to_step(base_htc, step_htc, **kw)
        print('   done.')

    if make_turb:
        turb_htc = base_htc.replace('.htc', '_turb.htc')
        print(f' Writing turbulence file to {turb_htc}...')
        base_to_turb(base_htc, turb_htc, **kw)
        print('   done.')

    if make_hs2:
        hs2_htc = base_htc.replace('.htc', '_hs2.htc').replace('../htc', '../')
        print(f'  Writing hawcstab2 file to {hs2_htc}...')
        base_to_hs2(base_htc, hs2_htc, **kw)
        print('   done.')
