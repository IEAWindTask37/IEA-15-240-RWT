'''
Update the DISCON.IN examples in the IEA-15MW repository using tuning .yamls in rosco conda environment

'''
import os
from ROSCO_toolbox.ofTools.fast_io.update_discons import update_discons


if __name__=="__main__":

    # directory references
    this_dir = os.path.dirname(os.path.abspath(__file__))
    of_dir = os.path.realpath(os.path.join(this_dir,'../..'))

    # paths relative to this OpenFAST directory
    map_rel = {
        'IEA-15-240-RWT-Monopile/ServoData/IEA15MW-Monopile.yaml': 'IEA-15-240-RWT-Monopile/ServoData/DISCON-Monopile.IN',
        'IEA-15-240-RWT-UMaineSemi/ServoData/IEA15MW-UMaineSemi.yaml': 'IEA-15-240-RWT-UMaineSemi/ServoData/DISCON-UMaineSemi.IN',
    }

    # Make paths absolute
    map_abs = {}
    for tune, test in map_rel.items():
        tune = os.path.join(of_dir,tune)
        map_abs[tune] = os.path.join(of_dir,test)

    # Make discons    
    update_discons(map_abs)
