import numpy as np
import pandas as pd

# Read tower dimensions definition

tower_def=pd.read_csv('umaine_tower_def.dat',sep=' ')
print(tower_def)

#   write st-file

E   =  200.0e+09
G   =   79.3e+09
rho = 7850.0e+00

nfmt = 23-10
def get_st_line(r,D,t,E,G,rho):
    global nfmt
    A  = np.pi/4.0 *(D**2 - (D - 2.0*t*0.001)**2)
    Ix = np.pi/64.0*(D**4 - (D - 2.0*t*0.001)**4)
    ri_x = np.sqrt(Ix/A)
    fmt = '{:' + str(nfmt) + '.' + str(nfmt-8) + 'e}'
    dline = {
        'r'     : fmt.format(r), 
        'm'     : fmt.format(rho*A),
        'x_cg'  : fmt.format(0.0),
        'y_cg'  : fmt.format(0.0),
        'ri_x'  : fmt.format(ri_x),
        'ri_y'  : fmt.format(ri_x),
        'x_sh'  : fmt.format(0.0),
        'y_sh'  : fmt.format(0.0),
        'E'     : fmt.format(E),
        'G'     : fmt.format(G),
        'I_x'   : fmt.format(Ix),
        'I_y'   : fmt.format(Ix),
        'K'     : fmt.format(2.0*Ix),
        'k_x'   : fmt.format(0.5),
        'k_y'   : fmt.format(0.5),
        'A'     : fmt.format(A),
        'pitch' : fmt.format(0.0),
        'x_e'   : fmt.format(0.0),
        'y_e'   : fmt.format(0.0)
        }
    return dline


files = {'umaine_tower_st.dat'}
for file in files:
    f = open(file,'w')
    f.write('1  number of sets, Nset\n')
    f.write('#1\n')
    dline = get_st_line(0.0,1.0,1.0,0.0,0.0,0.0) # dummy line to get keys
    for key in dline:
        fmt = '{:>' + str(nfmt) + 's}'
        f.write(fmt.format(key))
    f.write('\n')    
    f.write('$1 {:d}\n'.format(len(tower_def)))
    for i in np.arange(len(tower_def)):
        if True:
            origin = tower_def.at[0,'loc[m]']
            dline = get_st_line(tower_def.at[i,'loc[m]'] - origin, 
                                tower_def.at[i,'D[m]'  ],
                                tower_def.at[i,'t[mm]' ],
                                E,G,rho)
            for key in dline:
                f.write(dline[key])
            f.write('\n')
    f.close() 