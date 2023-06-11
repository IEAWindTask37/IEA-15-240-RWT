"""Make soil file from prescribed parameters.

We include lateral, axial, and torsional stiffness, but not rocking.

TODO: Fix this to return forces, not force-per-length.
"""
from datetime import date
from pathlib import Path

import numpy as np


G_soil = 140e6  # soil shear modulus [Pa]
nu_soil = 0.4  # soil poisson ratio [-]
r0 = 10/2  # outer radius of the soil [m]
defl = 1 # deflections to put in soil file [m or rad]
soil_path = Path(__file__).parents[1] / 'soil/IEA_15MW_Soil.dat'  # directory of hawc2 model (one level above)

zGs = np.arange(30, 76, 5)  # global z coordinates
hs = zGs - zGs.min()  # depth under the soil

# vertical distributed stiffness [N/m/m]
eta_z = 1 + 0.6*(1 - nu_soil) * hs/r0
k_zs = 4*G_soil*r0 / (1 - nu_soil) * eta_z

# horizontal (lateral) distributed stiffness [N/m/m]
eta_x = 1 + 0.55*(2 - nu_soil) * hs/r0
k_xs = 32*(1 - nu_soil)*G_soil*r0 / (7 - 8*nu_soil) * eta_x

# torsional stiffness [Nm/rad/m]
k_phis = 16*G_soil*r0**3 / 3 * np.ones_like(hs)

# wave the soil file
with open(soil_path, 'w', encoding='utf-8') as f:
    f.write(f'Distributed soil stiffness for IEA 15 MW Monopile created on {date.today().strftime("%d-%b-%Y")}\n')
    spring_vals = (('lateral', k_xs), ('axial', k_zs), ('rotation_z', k_phis))
    defls = np.array([0, defl])
    for i_s, (stiffname, stiffs) in enumerate(spring_vals):
        # set number, type, nrow/ndefl, and defls
        f.write(f'#{i_s + 1}\n')
        f.write(stiffname + '\n')
        f.write(f'{zGs.size} {defls.size} nrow ndefl\n')
        f.write(np.array2string(defls)[1:-1] + ' x1 x2 x3 ..... [m]\n')
        # loop over zGs
        for zG, k in zip(zGs, stiffs):
            force_mom = k * defl * 1e-3  # force = k * x OR mom = km * theta; kN or kNm
            f.write(f'{zG}  0  {force_mom:8.0f}\n')
