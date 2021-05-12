# -*- coding: utf-8 -*-
"""
Generate .htc file to use as base for DLBs generation with the script
make_dlbs.py. Script by thea, inspired by Anders's.

"""
import pandas as pd

# Set general parameters (dlbs will be created afterwards, so no need for precision)
filename = 'IEA_15MW_RWT_UMaine'
wsp = 8
sim_time = 900
ini_time = 300
dx = 8192
Iref = 0.14
tint = Iref * (0.75 * wsp + 5.6) / wsp
mann_dx = (sim_time - ini_time) * wsp / dx
on = ''
off = ';'

# define dictionary to build the htc
dict_data = {'case':            filename,
             'flnm':            filename,
             'wtg':             on,
             'floater':         on,
             'mooring':         on,
             'mooring_sys':     'htc/template/mooring_sys.inc',
             'sim_time':        sim_time,
             'time_step':       0.01,
             'ini_time':        ini_time,
             'info':            off,
             'output':          on,
             'hydro':           on,
             'qtf':             off,
             'wave':            off,
             'wind':            on,
             'control':         on,
             'control_rosco':   off,
             'anim':            off,
             'damper':          on,
             'extforce':        off,
             'shear_format':    3,
             'shear_exp':       0.14,
             'turb_format':     1,
             'tower_shadow':    3,
             'wsp':             wsp,
             'tint':            tint,
             'seed':            1001,
             'Mann_dx':         mann_dx
}


htc_out = f'./{filename}.htc'
data_in = pd.DataFrame(dict_data, index = [0])
brackets = '[]'

fin  = open('./template/template.htc', 'r')
fout =  open(htc_out, 'w')
for line in fin.readlines():
    for tag in list(data_in.keys()):
        if pd.isna(data_in.at[0, tag]):         # if the tag content is empty, replace brackets with empty (uncomment)
            srep = ''
        else:
            srep = str(data_in.at[0, tag])      # if the tag has a value, replace brackets with value
        line = line.replace(brackets[0] + tag + brackets[1], srep)
    fout.write(line)
fin.close()
fout.close()
