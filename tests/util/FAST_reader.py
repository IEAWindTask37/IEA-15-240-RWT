import os, re, copy
import numpy as np
from functools import reduce
import operator

def bool_read(text):
    # convert true/false strings to boolean
    if 'default' in text.lower():
        return str(text)
    else:
        if text.lower() == 'true':
            return True
        else:
            return False

def float_read(text):
    # return float with error handing for "default" values
    if 'default' in text.lower():
        return str(text)
    else:
        try:
            return float(text)
        except:
            return str(text)

def int_read(text):
    # return int with error handing for "default" values
    if 'default' in text.lower():
        return str(text)
    else:
        try:
            return int(text)
        except:
            return str(text)

class InputReader_OpenFAST(object):
    """ OpenFAST input file reader """

    def __init__(self):

        self.FAST_directory = None   # Path to fst directory files
        self.fst_vt         = {}
        self.fst_vt['Fst']  = {}
        self.fst_vt['ElastoDyn'] = {}
        self.fst_vt['ElastoDynBlade'] = {}
        self.fst_vt['BeamDyn'] = {}
        self.fst_vt['BeamDynBlade'] = {}

    def set_outlist(self, vartree_head, channel_list):
        """ Loop through a list of output channel names, recursively set them to True in the nested outlist dict """

        # given a list of nested dictionary keys, return the dict at that point
        def get_dict(vartree, branch):
            return reduce(operator.getitem, branch, vartree_head)
        # given a list of nested dictionary keys, set the value of the dict at that point
        def set_dict(vartree, branch, val):
            get_dict(vartree, branch[:-1])[branch[-1]] = val
        # recursively loop through outlist dictionaries to set output channels
        def loop_dict(vartree, search_var, branch):
            for var in vartree.keys():
                branch_i = copy.copy(branch)
                branch_i.append(var)
                if type(vartree[var]) is dict:
                    loop_dict(vartree[var], search_var, branch_i)
                else:
                    if var == search_var:
                        set_dict(vartree_head, branch_i, True)

        # loop through outchannels on this line, loop through outlist dicts to set to True
        for var in channel_list:
            var = var.replace(' ', '')
            loop_dict(vartree_head, var, [])

    def read_ElastoDynBlade(self, blade_file):
        # ElastoDyn v1.00 Blade Input File
        # Currently no differences between FASTv8.16 and OpenFAST.

        f = open(blade_file)
        # print blade_file
        f.readline()
        f.readline()
        f.readline()
        
        # Blade Parameters
        self.fst_vt['ElastoDynBlade']['NBlInpSt'] = int(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['BldFlDmp1'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['BldFlDmp2'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['BldEdDmp1'] = float_read(f.readline().split()[0])
        
        # Blade Adjustment Factors
        f.readline()
        self.fst_vt['ElastoDynBlade']['FlStTunr1'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['FlStTunr2'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['AdjBlMs'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['AdjFlSt'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['AdjEdSt'] = float_read(f.readline().split()[0])
        
        # Distrilbuted Blade Properties
        f.readline()
        f.readline()
        f.readline()
        self.fst_vt['ElastoDynBlade']['BlFract'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        self.fst_vt['ElastoDynBlade']['PitchAxis'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        self.fst_vt['ElastoDynBlade']['StrcTwst'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        self.fst_vt['ElastoDynBlade']['BMassDen'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        self.fst_vt['ElastoDynBlade']['FlpStff'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        self.fst_vt['ElastoDynBlade']['EdgStff'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']

        for i in range(self.fst_vt['ElastoDynBlade']['NBlInpSt']):
            data = f.readline().split()          
            self.fst_vt['ElastoDynBlade']['BlFract'][i]  = float_read(data[0])
            self.fst_vt['ElastoDynBlade']['PitchAxis'][i]  = float_read(data[1])
            self.fst_vt['ElastoDynBlade']['StrcTwst'][i]  = float_read(data[2])
            self.fst_vt['ElastoDynBlade']['BMassDen'][i]  = float_read(data[3])
            self.fst_vt['ElastoDynBlade']['FlpStff'][i]  = float_read(data[4])
            self.fst_vt['ElastoDynBlade']['EdgStff'][i]  = float_read(data[5])

        f.readline()
        self.fst_vt['ElastoDynBlade']['BldFl1Sh'] = [None] * 5
        self.fst_vt['ElastoDynBlade']['BldFl2Sh'] = [None] * 5        
        self.fst_vt['ElastoDynBlade']['BldEdgSh'] = [None] * 5
        for i in range(5):
            self.fst_vt['ElastoDynBlade']['BldFl1Sh'][i]  = float_read(f.readline().split()[0])
        for i in range(5):
            self.fst_vt['ElastoDynBlade']['BldFl2Sh'][i]  = float_read(f.readline().split()[0])            
        for i in range(5):
            self.fst_vt['ElastoDynBlade']['BldEdgSh'][i]  = float_read(f.readline().split()[0])        

        f.close()

    def read_BeamDyn(self, bd_file):
        # BeamDyn Input File
        f = open(bd_file)
        f.readline()
        f.readline()
        f.readline()
        # ---------------------- SIMULATION CONTROL --------------------------------------
        self.fst_vt['BeamDyn']['Echo']             = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['QuasiStaticInit']  = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['rhoinf']           = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['quadrature']       = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['refine']           = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['n_fact']           = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['DTBeam']           = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['load_retries']     = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['NRMax']            = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['stop_tol']         = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['tngt_stf_fd']      = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['tngt_stf_comp']    = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['tngt_stf_pert']    = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['tngt_stf_difftol'] = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['RotStates']        = bool_read(f.readline().split()[0])
        f.readline()
        #---------------------- GEOMETRY PARAMETER --------------------------------------
        self.fst_vt['BeamDyn']['member_total']     = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['kp_total']         = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['members']          = []
        for i in range(self.fst_vt['BeamDyn']['member_total']):
            ln = f.readline().split()
            n_pts_i                   = int(ln[1])
            member_i                  = {}
            member_i['kp_xr']         = [None]*n_pts_i
            member_i['kp_yr']         = [None]*n_pts_i
            member_i['kp_zr']         = [None]*n_pts_i
            member_i['initial_twist'] = [None]*n_pts_i
            f.readline()
            f.readline()
            for j in range(n_pts_i):
                ln = f.readline().split()
                member_i['kp_xr'][j]          = float(ln[0])
                member_i['kp_yr'][j]          = float(ln[1])
                member_i['kp_zr'][j]          = float(ln[2])
                member_i['initial_twist'][j]  = float(ln[3])

            self.fst_vt['BeamDyn']['members'].append(member_i)
        #---------------------- MESH PARAMETER ------------------------------------------
        f.readline()
        self.fst_vt['BeamDyn']['order_elem']  = int_read(f.readline().split()[0])
        #---------------------- MATERIAL PARAMETER --------------------------------------
        f.readline()
        self.fst_vt['BeamDyn']['BldFile']     = f.readline().split()[0].replace('"','').replace("'",'')
        #---------------------- PITCH ACTUATOR PARAMETERS -------------------------------
        f.readline()
        self.fst_vt['BeamDyn']['UsePitchAct'] = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['PitchJ']      = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['PitchK']      = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['PitchC']      = float_read(f.readline().split()[0])
        #---------------------- OUTPUTS -------------------------------------------------
        f.readline()
        self.fst_vt['BeamDyn']['SumPrint']    = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['OutFmt']      = f.readline().split()[0][1:-1]
        self.fst_vt['BeamDyn']['NNodeOuts']   = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['OutNd']       = [idx.strip() for idx in f.readline().split('NNodeOuts')[0].split(',')]
        # BeamDyn Outlist
        f.readline()
        data = f.readline()
        while data.split()[0] != 'END':
            channels = data.split('"')
            channel_list = channels[1].split(',')
            self.set_outlist(self.fst_vt['outlist']['BeamDyn'], channel_list)
            data = f.readline()

        f.close()

        self.read_BeamDynBlade()

    def read_BeamDynBlade(self):
        # BeamDyn Blade

        beamdyn_blade_file = os.path.join(self.FAST_directory, self.fst_vt['BeamDyn']['BldFile'])
        f = open(beamdyn_blade_file)
        
        f.readline()
        f.readline()
        f.readline()
        #---------------------- BLADE PARAMETERS --------------------------------------
        self.fst_vt['BeamDynBlade']['station_total'] = int_read(f.readline().split()[0])
        self.fst_vt['BeamDynBlade']['damp_type']     = int_read(f.readline().split()[0])
        f.readline()
        f.readline()
        f.readline()
        #---------------------- DAMPING COEFFICIENT------------------------------------
        ln = f.readline().split()
        self.fst_vt['BeamDynBlade']['mu1']           = float(ln[0])
        self.fst_vt['BeamDynBlade']['mu2']           = float(ln[1])
        self.fst_vt['BeamDynBlade']['mu3']           = float(ln[2])
        self.fst_vt['BeamDynBlade']['mu4']           = float(ln[3])
        self.fst_vt['BeamDynBlade']['mu5']           = float(ln[4])
        self.fst_vt['BeamDynBlade']['mu6']           = float(ln[5])
        f.readline()
        #---------------------- DISTRIBUTED PROPERTIES---------------------------------
        
        self.fst_vt['BeamDynBlade']['radial_stations'] = np.zeros((self.fst_vt['BeamDynBlade']['station_total']))
        self.fst_vt['BeamDynBlade']['beam_stiff']      = np.zeros((self.fst_vt['BeamDynBlade']['station_total'], 6, 6))
        self.fst_vt['BeamDynBlade']['beam_inertia']    = np.zeros((self.fst_vt['BeamDynBlade']['station_total'], 6, 6))
        for i in range(self.fst_vt['BeamDynBlade']['station_total']):
            self.fst_vt['BeamDynBlade']['radial_stations'][i]  = float_read(f.readline().split()[0])
            for j in range(6):
                self.fst_vt['BeamDynBlade']['beam_stiff'][i,j,:] = np.array([float(val) for val in f.readline().strip().split()])
            f.readline()
            for j in range(6):
                self.fst_vt['BeamDynBlade']['beam_inertia'][i,j,:] = np.array([float(val) for val in f.readline().strip().split()])
            f.readline()

        f.close()
