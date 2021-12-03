import numpy as np
import pyFAST.converters.beamdyn as bd
from shutil import copyfile
import matplotlib.pyplot as plt


suffix='beamdyn_c2' 

AEG = np.loadtxt('/Users/pbortolo/work/2_openfast/stability/BDstab/IEA15/OF2H2/repo_files/IEA15master_noFPM_flex.dat', skiprows = 8)

grid = np.array([0.   , 0.01 , 0.02 , 0.03 , 0.04 , 0.05, 0.075, 
       0.1  , 0.15 , 0.2  , 0.25 , 0.3  , 0.35 ,
       0.4  , 0.45 , 0.5  , 0.55 , 0.6  , 0.65 ,
       0.7  , 0.75 , 0.8  , 0.85 , 0.9  , 0.95 , 1.])
E = np.interp(grid, AEG[:,0]/AEG[-1,0], AEG[:, 8])
G = np.interp(grid, AEG[:,0]/AEG[-1,0], AEG[:, 9])
A = np.interp(grid, AEG[:,0]/AEG[-1,0], AEG[:, 15])


# --- BeamDynToHawc2
htc_template   = '/Users/pbortolo/work/3_projects/5_IEAtask37/IEA-15-240-RWT/HAWC2/IEA-15-240-RWT-FixedBottom/htc/IEA_15MW_RWT.htc'
BD_mainfile    = '/Users/pbortolo/work/3_projects/5_IEAtask37/IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT/IEA-15-240-RWT_BeamDyn_c2.dat'
BD_bladefile   = '/Users/pbortolo/work/3_projects/5_IEAtask37/IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT/IEA-15-240-RWT_BeamDyn_blade_c2.dat'

H2_htcfile_new = '/Users/pbortolo/work/3_projects/5_IEAtask37/IEA-15-240-RWT/HAWC2/IEA-15-240-RWT-FixedBottom/htc/IEA15_{}.htc'.format(suffix+'_FPM')
H2_stfile      = '/Users/pbortolo/work/3_projects/5_IEAtask37/IEA-15-240-RWT/HAWC2/IEA-15-240-RWT-FixedBottom/data/blade_{}.st'.format(suffix+'_FPM')
copyfile(htc_template,  H2_htcfile_new) # Backup template
df_c2, df_st = bd.beamDynToHawc2(BD_mainfile, BD_bladefile, H2_htcfile_new, H2_stfile, 'blade1', A=A, E=E, G=G, theta_p_in = None, FPM=True, verbose=True)

H2_htcfile_new = '/Users/pbortolo/work/3_projects/5_IEAtask37/IEA-15-240-RWT/HAWC2/IEA-15-240-RWT-FixedBottom/htc/IEA15_{}.htc'.format(suffix+'_noFPM')
H2_stfile      = '/Users/pbortolo/work/3_projects/5_IEAtask37/IEA-15-240-RWT/HAWC2/IEA-15-240-RWT-FixedBottom/data/blade_{}.st'.format(suffix+'_noFPM')
copyfile(htc_template,  H2_htcfile_new) # Backup template
df_c2, df_st = bd.beamDynToHawc2(BD_mainfile, BD_bladefile, H2_htcfile_new, H2_stfile, 'blade1', A=A, E=E, G=G, theta_p_in = None, FPM=False, verbose=True)
