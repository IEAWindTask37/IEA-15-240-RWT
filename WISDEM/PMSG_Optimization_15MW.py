"""PMSG_Outer_rotor.py

Created by Latha Sethuraman
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on {Structural mass in direct-drive permanent magnet electrical generators by
McDonald,A.S. et al. IET Renewable Power Generation(2008),2(1):3 http://dx.doi.org/10.1049/iet-rpg:20070071 """

import pandas 
import numpy as np
from openmdao.api import ExplicitComponent, Group, Problem, IndepVarComp,ExecComp,pyOptSparseDriver
from openmdao.drivers.scipy_optimizer import ScipyOptimizer,ScipyOptimizeDriver
from pulp import *
from numpy import array,float,min,sign
from fractions import gcd
from math import pi, cos, sqrt, radians, sin, sinh, cosh, exp, log10, log, tan, atan,ceil,gcd
from winding_design import winding_factor



#Assign values to universal constants
        
B_r                         =   1.279                 # Tesla remnant flux density
E                           =   2e11                # N/m^2 young's modulus
ratio                       =   0.8                 # ratio of magnet width to pole pitch(bm/self.tau_p)
mu_0                        =   pi*4e-7             # permeability of free space

mu_r                        =   1.06                # relative permeability
cofi                        =   0.85                 # power factor

#Assign values to design constants
h_0                         =   0.005				# Slot opening height
h_1                         =   0.004 				# Slot wedge height
m                           =   3                   # no of phases
#b_s_tau_s=0.45 							 # slot width to slot pitch ratio
k_sfil                      =   0.65								 # Slot fill factor
P_Fe0h =4			               #specific hysteresis losses W/kg @ 1.5 T 
P_Fe0e =1			               #specific hysteresis losses W/kg @ 1.5 T


k_fes=0.8											# Iron fill factor

#Assign values to universal constants

g1         =   9.8106              # m/s**2 acceleration due to gravity
E          =   2e11                # Young's modulus
phi        =   90*2*pi/360         # tilt angle (rotor tilt -90 degrees during transportation)
v          =    0.3                # Poisson's ratio
G          =   79.3e9



def shell_constant(R,t,l,x,v):
    
    Lambda     = (3*(1-v**2)/(R**2*t**2))**0.25
    D          = E*t**3/(12*(1-v**2))
    C_14       = (sinh(Lambda*l))**2+ (sin(Lambda*l))**2
    C_11       = (sinh(Lambda*l))**2- (sin(Lambda*l))**2
    F_2        = cosh(Lambda*x)*sin(Lambda*x) + sinh (Lambda*x)* cos(Lambda*x)
    C_13       = cosh(Lambda*l)*sinh(Lambda*l) - cos(Lambda*l)* sin(Lambda*l)
    F_1        = cosh(Lambda*x)*cos(Lambda*x)
    F_4        = cosh(Lambda*x)*sin(Lambda*x)-sinh(Lambda*x)*cos(Lambda*x)
    
    return D,Lambda,C_14,C_11,F_2,C_13,F_1,F_4
        
def plate_constant(a,b,v,r_o,t):
    
    D          = E*t**3/(12*(1-v**2))
    C_2        = 0.25*(1-(b/a)**2*(1+2*log(a/b)))
    C_3        = 0.25*(b/a)*(((b/a)**2+1)*log(a/b)+(b/a)**2 -1)
    C_5        = 0.5*(1-(b/a)**2)
    C_6        = 0.25*(b/a)*((b/a)**2-1+2*log(a/b))
    C_8        = 0.5*(1+v+(1-v)*(b/a)**2)
    C_9        = (b/a)*(0.5*(1+v)*log(a/b)+0.25*(1-v)*(1-(b/a)**2))
    L_11       = (1/64)*(1+4*(r_o/a)**2-5*(r_o/a)**4-4*(r_o/a)**2*(2+(r_o/a)**2)*log(a/r_o))
    L_17       = 0.25*(1-0.25*(1-v)*((1-(r_o/a)**4)-(r_o/a)**2*(1+(1+v)*log(a/r_o))))
            
    return D,C_2,C_3,C_5,C_6,C_8,C_9,L_11,L_17


class PMSG_active(ExplicitComponent):

    """ Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator. """
    
    def setup(self):
    
        self.add_input('r_g',0.0, units ='m', desc='airgap radius ')
        self.add_input('l_s',0.0, units ='m', desc='Stator core length ')
        self.add_input('h_s',0.0, units ='m', desc='Yoke height ')
        self.add_input('P_rated', units ='W', desc='Machine rating')
        self.add_input('N_nom', 0.0, units = 'rpm', desc='rated speed')
        self.add_input('T_rated',0.0, units = 'N*m', desc='Rated torque ')
        self.add_input('h_m',0.0, units ='m',desc='magnet height')
        self.add_input('h_ys',0.0, units ='m', desc='Yoke height')
        self.add_input('h_yr',0.0, units ='m', desc='rotor yoke height')
        self.add_input('J_s',0.0, units ='A/(mm*mm)', desc='Stator winding current density')
        self.add_input('N_c',0.0, desc='Number of turns per coil')
        self.add_input('b',0.0, desc='Slot pole combination')
        self.add_input('c',0.0, desc='Slot pole combination')
        self.add_input('p',0.0, desc='Pole pairs ')
        self.add_input('Sigma',0.0,units='N/m**2',desc='Shear Stress')

        
        # Material properties
        self.add_input('rho_Fes',0.0, units='kg/(m**3)',desc='Structural Steel density')
        self.add_input('rho_Fe',0.0, units='kg/(m**3)', desc='Electrical Steel density ')
        self.add_input('rho_Copper',0.0,units='kg/(m**3)', desc='Copper density')
        self.add_input('rho_PM',0.0,units ='kg/(m**3)', desc='Magnet density ')
        self.add_input('rho_Cu',0.0,units ='ohm*m', desc='Copper resistivity ')
        
        # PMSG_structrual inputs
        self.add_input('Structural_mass_rotor',0.0,units='kg',desc='Structural Mass')
        self.add_input('Structural_mass_stator',0.0,units='kg',desc='Structural Mass')
        self.add_input('h_sr',0.0,units='m',desc='Structural Mass')
        
        # Magnetic loading
        
        self.add_input('B_tmax',0.0, desc='Peak Teeth flux density')
        self.add_output('B_symax',0.0,desc='Peak Stator Yoke flux density B_ymax')
        self.add_output('B_rymax',0.0, desc='Peak Rotor yoke flux density')
        self.add_output('B_smax',0.0,desc='Peak Stator flux density')
        self.add_output('B_pm1', 0.0,desc='Fundamental component of peak air gap flux density')
        self.add_output('B_g', 0.0, desc='Peak air gap flux density ')
        self.add_output('tau_p',0.0, units ='m',desc='Pole pitch')
        self.add_output('p1',0.0,desc='Pole pairs')
        self.add_output('q',0.0, units ='N/m**2',desc='Normal stress')
        self.add_output('g',0.0,units='m',desc='Air gap length')
        
        #Stator design
        self.add_output('N_s', 0.0, desc='Number of turns in the stator winding')
        self.add_output('b_s',0.0, units ='m', desc='slot width')
        self.add_output('b_t',0.0, units = 'm',desc='tooth width')
        self.add_output('h_t',0.0, units = 'm', desc ='tooth height')
        self.add_output('A_Cuscalc',0.0, units ='mm**2', desc='Conductor cross-section')
        self.add_output('tau_s', 0.0, units='m',desc='Slot pitch ')
        
        #Rotor magnet dimension
        self.add_output('b_m',0.0, units ='m',desc='magnet width')
        
        
        # Electrical performance
        self.add_output('E_p',0.0, units ='V', desc='Stator phase voltage')
        self.add_output('f',0.0, units ='Hz', desc='Generator output frequency')
        self.add_output('I_s', 0.0, units ='A', desc='Generator output phase current')
        self.add_output('R_s',0.0, units ='ohm',desc='Stator resistance')
        self.add_output('L_s',0.0,desc='Stator synchronising inductance')
        self.add_output('A_1',0.0, units='A/m', desc='Electrical loading')
        self.add_output('J_actual',0.0, units ='A/m**2', desc='Current density')
        self.add_output('Test',0.0, desc='magnet width')
        self.add_output('T_e',0.0,units='N*m',desc='Electromagnetic torque')
        
        
        
        # Objective functions
        self.add_output('Mass',0.0, units='kg',desc='Actual mass')
        self.add_output('K_rad', desc='Aspect ratio')
        self.add_output('Losses',desc='Total power losses')
        self.add_output('gen_eff', desc='Generator efficiency')
        
        
       
        # Other parameters
        self.add_output('R_out',0.0,units='m', desc='Outer radius')
        self.add_output('S',0.0,desc='Stator slots')
        self.add_output('Slot_aspect_ratio',0.0,desc='Slot aspect ratio')
        
        # Mass Outputs
        self.add_output('mass_PM',0.0,units = 'kg', desc='Magnet mass')
        self.add_output('Copper',0.0, units ='kg', desc='Copper Mass')
        self.add_output('Iron',0.0, units = 'kg', desc='Electrical Steel Mass')
        self.add_output('Mass_tooth_stator',0.0, units = 'kg', desc='Teeth and copper mass')
        self.add_output('Mass_yoke_rotor',0.0, units = 'kg', desc='yoke mass')
        self.add_output('Mass_yoke_stator',0.0, units = 'kg', desc='yoke mass')
        
        self.add_output('Structural_mass',units='kg',desc='Total Structural Mass')

        self.add_output('I',np.zeros(3),units='kg*m**2',desc='Structural Mass')
        
        
        self.declare_partials('*','*',method='fd')
        
        
        
        
    def compute(self, inputs, outputs):
        
        
        ###################################################### Electromagnetic design#############################################
        
        outputs['K_rad']=inputs['l_s'][0]/(2* inputs['r_g'][0])							# Aspect ratio
        
        # rated torque
        
        # Calculating air gap length
        
        dia				            =  2* inputs['r_g'][0]              # air gap diameter

        outputs['g']                =  0.001*dia                        # air gap length
        r_m     	                =  inputs['r_g']                    #magnet radius
        r_s				            =  inputs['r_g']-outputs['g']                  #Stator outer radius
        b_so			            =  2*outputs['g']                              # Slot opening

        outputs['p1']               =  inputs['p'] #ceil(inputs['p'])
        #outputs['p1']               =  ceil(inputs['p'][0]/ 5.) * 5
        
        outputs['tau_p']            =  (pi*dia/(2*outputs['p1']))	        # pole pitch
        
        
        #Calculating winding factor
        
        Slot_pole                   = inputs['b']/inputs['c']
        
        outputs['S']                =  Slot_pole*2*outputs['p1']*m
        
        outputs['Test']             =  outputs['S'][0]/(m*gcd(int(outputs['S'][0]),int(outputs['p1'][0])))
    
        if outputs['Test'][0].is_integer():
            
            k_w				        =  winding_factor(int(outputs['S'][0]),inputs['b'][0],inputs['c'][0],int(outputs['p1'][0]),m)
            outputs['b_m']          = ratio*outputs['tau_p']								 # magnet width
            alpha_p		            =  pi/2*ratio
            outputs['tau_s']	    =   pi*(dia-2*outputs['g'])/outputs['S']
            b_s_tau_s               = 0.45
            
            # Calculating Carter factor for statorand effective air gap length
            
            gamma			        =  4/pi*(b_so/2/(outputs['g']+inputs['h_m']/mu_r)*atan(b_so/2/(outputs['g']+ inputs['h_m']/mu_r))-log(sqrt(1+(b_so/2/(outputs['g']+inputs['h_m']/mu_r))**2)))
            k_C				        =  outputs['tau_s']/(outputs['tau_s']-gamma*(outputs['g']+inputs['h_m']/mu_r))   # carter coefficient
            g_eff			        =  k_C*(outputs['g']+ inputs['h_m']/mu_r)
            
           
            # angular frequency in radians
            om_m			        =  2*pi*inputs['N_nom']/60
            om_e			        =  outputs['p1']*om_m
            outputs['f']                =  om_e/2/pi					# outout frequency
            
            # Calculating magnetic loading
            outputs['B_pm1'] 		=  B_r*inputs['h_m'][0]/mu_r/(g_eff)
            outputs['B_g']          =  B_r*inputs['h_m'][0]/(mu_r*g_eff)*(4/pi)*sin(alpha_p)
           
            outputs['B_symax']      =  outputs['B_pm1']* outputs['b_m']/(2*inputs['h_ys'])*k_fes
            outputs['B_rymax']      =  outputs['B_pm1']*outputs['b_m']*k_fes/(2*inputs['h_yr'])
            outputs['b_t']          =  outputs['B_pm1']*outputs['tau_s']/inputs['B_tmax']
            N_c                     =  2    # Number of turns per coil
            #self.b_t	= self.tau_s-self.b_s 					#slot width
            outputs['N_s']          =   outputs['S']*2.0/3*N_c									# Stator turns per phase
            #outputs['N_s']          =   outputs['p1']*inputs['Slot_pole']*N_c*3                  #2*m*p*q
            
            
            outputs['q']            = (outputs['B_g'])**2/2/mu_0
            
            #self.b_t	            =  self.tau_s-(self.b_s)          		#tooth width
            
            # Stator winding length ,cross-section and resistance
            l_Cus			        = (2*(inputs['l_s']+pi/4*(outputs['tau_s']+outputs['b_t'])))  # length of a turn
            
            # Calculating no-load voltage induced in the stator
            outputs['E_p']	        = 2*(outputs['N_s'])*inputs['l_s']*r_s*k_w*om_m*outputs['B_g']/sqrt(2)

            #
            Z                       =(inputs['P_rated'][0]/(m*outputs['E_p'][0]))
            
            # Calculating leakage inductance in  stator
            V_1                     =outputs['E_p']/1.1
            I_n			            = inputs['P_rated'][0]/3/cofi/V_1
            J_s         =   6.0
            
            outputs['A_Cuscalc']	= I_n/J_s
            
            A_slot                  = 2*N_c*outputs['A_Cuscalc']*(10**-6)/k_sfil
            
            
            
            tau_s_new               =pi*(dia-2*outputs['g']-2*h_1-2*h_0)/outputs['S']
            
            b_s2		            =tau_s_new-outputs['b_t'] # Slot top width
            
            b_s1		            =sqrt(b_s2**2-4*pi*A_slot/outputs['S'])
            
            #b_s1		            =(2*A_slot-b_s2*self.h_s)/self.h_s
            
            print (b_s1,b_s2)
            
            outputs['b_s']	        =(b_s1+b_s2)*0.5
            
            

            #b_t_self.tau_s         =A[4]
            
            outputs['Slot_aspect_ratio']= inputs['h_s']/outputs['b_s']
            
            N_coil                      =   2*outputs['S'][0]
            
            P_s                         =   mu_0*(inputs['h_s'][0]/3/outputs['b_s'][0] +h_1*2/(b_s2+b_so)+h_0/b_so)    #Slot permeance function
            
            L_ssigmas                   =   outputs['S']/3*4*N_c**2*inputs['l_s']*P_s  #slot leakage inductance

            L_ssigmaew                  =   N_coil*N_c**2*mu_0*outputs['tau_s']*log((0.25*pi*outputs['tau_s']**2)/(0.5*inputs['h_s']*outputs['b_s']))     #end winding leakage inductance
            L_aa                        =   2*pi/3*(N_c**2*mu_0*inputs['l_s']*r_s/g_eff)
            
            L_ab                        =   0
            
            L_m                         =   L_aa
            
            L_ssigma	                = (L_ssigmas+L_ssigmaew)
            
            outputs['L_s']              = L_m+L_ssigma
            
            G                           =  (outputs['E_p']**2-(om_e*outputs['L_s'])**2*Z**2)
            
            # Calculating stator current and electrical loading
            
            I_s              = sqrt(Z**2+(((outputs['E_p']-G**0.5)/(om_e*outputs['L_s'])**2)**2))
            outputs['A_1']              =inputs['Sigma']/(0.5*outputs['B_g'])
            
           
            
            outputs['I_s']               = (pi*(dia-2*outputs['g']))*outputs['A_1']/(6*outputs['N_s'])/2**0.5
            
            
            #outputs['Sigma']            =0.5*outputs['B_g']*outputs['A_1']

            outputs['J_actual']         =   outputs['I_s']/(outputs['A_Cuscalc']*2**0.5)
            
                        
                      
            #outputs['R_s']          	= inputs['rho_Cu']*(outputs['N_s'])*l_Cus/(outputs['A_Cuscalc']*(10**-6))
            
            L_Cus                       = outputs['N_s']*l_Cus
            
            outputs['R_s']              =inputs['rho_Cu']*(outputs['N_s'])*l_Cus/(outputs['A_Cuscalc']*(10**-6))
            
            
            outputs['B_smax']           =   sqrt(2)*outputs['I_s']*mu_0/g_eff
            

            #(self.tau_s-b_so)*h_0+ (self.b_t+(self.tau_s-b_so))*0.5*h_1
            # Calculating Electromagnetically active mass
            
            wedge_area                       =   (outputs['b_s']*0.5-b_so*0.5)*(2*h_0+h_1)
            
            V_Cus 	                    =   m*L_Cus*(outputs['A_Cuscalc']*(10**-6))     # copper volume
            
            outputs['h_t']              =   (inputs['h_s']+h_1+h_0)
            
            V_Fest	                    =   inputs['l_s']*outputs['S']*(outputs['b_t']*(inputs['h_s']+h_1+h_0)+wedge_area)# volume of iron in stator tooth
            
            V_Fesy	                    =   inputs['l_s']*pi*((inputs['r_g']-outputs['g']-inputs['h_s']-h_1-h_0)**2-(inputs['r_g']-outputs['g']-inputs['h_s']-h_1-h_0-inputs['h_ys'])**2) # volume of iron in stator yoke
            
            V_Fery	                    =   inputs['l_s']*pi*((inputs['r_g']+inputs['h_m']+inputs['h_yr'])**2-(inputs['r_g']+inputs['h_m'])**2)
            
            outputs['Copper']		    =   V_Cus*inputs['rho_Copper']
            
            M_Fest	                    =   V_Fest*inputs['rho_Fe']    # Mass of stator tooth
            
            M_Fesy	                    =   V_Fesy*inputs['rho_Fe']    # Mass of stator yoke
            
            M_Fery	                    =   V_Fery*inputs['rho_Fe']    # Mass of rotor yoke
            
            outputs['Iron']     		=   M_Fest+M_Fesy+M_Fery
            
            
            outputs['mass_PM']          =   2*pi*(inputs['r_g']+ inputs['h_m'])*inputs['l_s']*inputs['h_m']*ratio*inputs['rho_PM']
            
            
            # Calculating Losses
            ##1. Copper Losses
            
            K_R                         =   1.0   # Skin effect correction co-efficient
            
            P_Cu		                =   m*(outputs['I_s']/2**0.5)**2*outputs['R_s']*K_R
            
            # Iron Losses ( from Hysteresis and eddy currents)
            P_Hyys	                    =   M_Fesy*(outputs['B_symax']/1.5)**2*(P_Fe0h*om_e/(2*pi*60)) # Hysteresis losses in stator yoke
            
            P_Ftys	                    =   M_Fesy*((outputs['B_symax']/1.5)**2)*(P_Fe0e*(om_e/(2*pi*60))**2) # Eddy losses in stator yoke
            
            P_Fesynom                   =   P_Hyys+P_Ftys
            
            P_Hyd                       =   M_Fest*(inputs['B_tmax']/1.5)**2*(P_Fe0h*om_e/(2*pi*60))  # Hysteresis losses in stator teeth
            
            P_Ftd                       =   M_Fest*(inputs['B_tmax']/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2) # Eddy losses in stator teeth
            
            P_Festnom                   =   P_Hyd+P_Ftd
                
            # additional stray losses due to leakage flux
            
            P_ad                        =0.2*(P_Hyys + P_Ftys + P_Hyd + P_Ftd )
            
            pFtm                        =300 # specific magnet loss
            
            P_Ftm                       =pFtm*2*outputs['p1']*outputs['b_m']*inputs['l_s']
            
            outputs['Losses']           =P_Cu+P_Festnom+P_Fesynom+P_ad+P_Ftm
            
            outputs['gen_eff']          =inputs['P_rated']*100/(inputs['P_rated']+outputs['Losses'])
            
            I_snom		                =outputs['gen_eff']*(inputs['P_rated']/m/outputs['E_p']/cofi) #rated current
            
            I_qnom		                =outputs['gen_eff']*inputs['P_rated']/(m* outputs['E_p'])
            
            X_snom		                =om_e*(L_m+L_ssigma)
            
            outputs['T_e']              =pi*inputs['r_g']**2*inputs['l_s']*2*inputs['Sigma']
 
            #self.TC3=R_st**2*l           # Evaluating Torque constraint for stator
            
            #self.Structural_mass=mass_stru_steel+(N_r*(R_1-self.R_o)*a_r*self.rho_Fes)
            
            Stator                      = M_Fesy+M_Fest+outputs['Copper'] #modified mass_stru_steel
            
            Rotor                       = M_Fery+outputs['mass_PM']  #modified (N_r*(R_1-self.R_o)*a_r*self.rho_Fes))
            
            outputs['Mass_tooth_stator']= M_Fest+outputs['Copper']
            
            outputs['Mass_yoke_rotor']= M_Fery
            
            outputs['Mass_yoke_stator']= M_Fesy
            
            outputs['Structural_mass']     = inputs['Structural_mass_rotor']+inputs['Structural_mass_stator']
            
            outputs['Mass']             = Stator+Rotor+outputs['Structural_mass']
            
            outputs['R_out']            = (dia+2*inputs['h_m']+2*inputs['h_yr']+2*inputs['h_sr'])*0.5
            
            print ('Optimised mass is:',outputs['Mass'], outputs['T_e']/1000)
            #print(outputs['A_1'],outputs['Sigma'])
            
            print (inputs['Structural_mass_stator'],inputs['Structural_mass_rotor'],M_Fest/1000)
        
        else:
            pass
            #outputs['A_1'] = 1e6
            #outputs['Slot_aspect_ratio']= 1e2
            #outputs['Mass']= 1e5
            #print ("Not an integer")
            			

class PMSG_rotor_inactive(ExplicitComponent):

    """ Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator. """
    
    def setup(self):
    
        self.add_input('R_sh',0.0, units ='m', desc='airgap radius ')

        
        self.add_input('y_sh', units ='W', desc='Machine rating')
        self.add_input('theta_sh', 0.0, units = 'rad', desc='slope')
        
        self.add_input('T_rated',0.0, units = 'N*m', desc='Rated torque ')
        self.add_input('r_g',0.0, units ='m', desc='air gap radius')
        self.add_input('h_m',0.0, units ='m', desc='Magnet height ')
        self.add_input('l_s',0.0, units ='m', desc='core length')
        self.add_input('q',0.0, units ='N/m**2', desc='Normal Stress')
        self.add_input('h_yr',0.0, units ='m', desc='Rotor yoke height ')
        
        self.add_input('u_allow_pcent',0.0,desc='Radial deflection as a percentage of air gap diameter')
        self.add_input('y_allow_pcent',0.0,desc='Radial deflection as a percentage of air gap diameter')
        self.add_input('z_allow_deg',0.0,units='deg',desc='Allowable torsional twist')
        
        
        # structural design variables
        self.add_input('t_r',0.0, units ='m', desc='Rotor disc thickness')
        self.add_input('h_sr',0.0, units ='m', desc='Yoke height ')
        
        self.add_input('K_rad', desc='Aspect ratio')
        
        # Material properties
        self.add_input('rho_Fes',0.0, units='kg/(m**3)',desc='Structural Steel density')
        self.add_input('rho_Fe',0.0, units='kg/(m**3)', desc='Electrical Steel density ')
        self.add_input('mass_PM',0.0,units ='kg', desc='Magnet density ')
        
        
        
        self.add_output('Structural_mass_rotor',0.0, units='kg', desc='Rotor mass (kg)')
        self.add_output('u_ar',0.0,units='m', desc='Radial deformation')
        self.add_output('y_ar',0.0, units ='m', desc='Axial deformation')
        self.add_output('twist_r', 0.0, units ='deg', desc='torsional twist')
               
        self.add_output('TC1r',0.0, desc='Torque constraint1-rotor')
        self.add_output('TC2r',0.0, desc='Torque constraint2-rotor')
        self.add_output('TC_test_r',0.0, desc='Torque constraint flag')
        
        self.add_output('u_allowable_r',0.0,units='m',desc='Allowable Radial deflection')
        self.add_output('y_allowable_r',0.0,units='m',desc='Allowable Radial deflection')
        self.add_input('Sigma',0.0,units='N/m**2',desc='Shear stress')

        
        
    def compute(self, inputs, outputs):
        
       
        # Radial deformation of rotor
        R               = inputs['r_g']+inputs['h_m']
        
        L_r             = inputs['l_s']+inputs['t_r']+0.025
        constants_x_0   = shell_constant(R,inputs['t_r'],L_r,0,v)
        constants_x_L   = shell_constant(R,inputs['t_r'],L_r,L_r,v)
        
       
        f_d_denom1      = R/(E*((R)**2-(inputs['R_sh'])**2))*((1-v)*R**2+(1+v)*(inputs['R_sh'])**2)
        f_d_denom2      = inputs['t_r']/(2*constants_x_0[0]*(constants_x_0[1])**3)*(constants_x_0[2]/(2*constants_x_0[3])*constants_x_0[4]-constants_x_0[5]/constants_x_0[3]*constants_x_0[6]-0.5*constants_x_0[7])
        
        f               = inputs['q']*(R)**2*inputs['t_r']/(E*(inputs['h_yr']+inputs['h_sr'])*(f_d_denom1+f_d_denom2))
        
        u_d             =f/(constants_x_L[0]*(constants_x_L[1])**3)*((constants_x_L[2]/(2*constants_x_L[3])*constants_x_L[4] -constants_x_L[5]/constants_x_L[3]*constants_x_L[6]-0.5*constants_x_L[7]))+inputs['y_sh']
        
 
        outputs['u_ar'] = (inputs['q']*(R)**2)/(E*(inputs['h_yr']+inputs['h_sr']))-u_d
                
        outputs['u_ar'] = abs(outputs['u_ar'] + inputs['y_sh'])
        
        outputs['u_allowable_r'] =2*inputs['r_g']/1000*inputs['u_allow_pcent']/100
        
        ###################################################### Electromagnetic design#############################################
        #return D,C_2,C_3,C_5,C_6,C_8,C_9,L_11,L_17
        # axial deformation of rotor
        W_back_iron     =  plate_constant(R+inputs['h_sr']+inputs['h_yr'],inputs['R_sh'],v,0.5*inputs['h_yr']+R,inputs['t_r'])
        W_ssteel        =  plate_constant(R+inputs['h_sr']+inputs['h_yr'],inputs['R_sh'],v,inputs['h_yr']+R+inputs['h_sr']*0.5,inputs['t_r'])
        W_mag           =  plate_constant(R+inputs['h_sr']+inputs['h_yr'],inputs['R_sh'],v,inputs['h_yr']+R-0.5*inputs['h_m'],inputs['t_r'])
        
        W_ir            =  inputs['rho_Fe']*g1*sin(phi)*(L_r-inputs['t_r'])*inputs['h_yr']
        y_ai1r          = -W_ir*(0.5*inputs['h_yr']+R)**4/(inputs['R_sh']*W_back_iron[0])*(W_back_iron[1]*W_back_iron[4]/W_back_iron[3]-W_back_iron[2])
        W_sr            =  inputs['rho_Fes']*g1*sin(phi)*(L_r-inputs['t_r'])*inputs['h_sr']
        y_ai2r          = -W_sr*(inputs['h_sr']*0.5+inputs['h_yr']+R)**4/(inputs['R_sh']*W_ssteel[0])*(W_ssteel[1]*W_ssteel[4]/W_ssteel[3]-W_ssteel[2])
        W_m            =  sin(phi)*inputs['mass_PM']/(2*pi*(R-inputs['h_m']*0.5))
        y_ai3r          = -W_m*(R-inputs['h_m'])**4/(inputs['R_sh']*W_mag[0])*(W_mag[1]*W_mag[4]/W_mag[3]-W_mag[2])
        
        w_disc_r        = inputs['rho_Fes']*g1*sin(phi)*inputs['t_r']
        
        a_ii            = R+inputs['h_sr']+inputs['h_yr']
        r_oii           = inputs['R_sh']
        M_rb            = -w_disc_r*a_ii**2/W_ssteel[5]*(W_ssteel[6]*0.5/(a_ii*inputs['R_sh'])*(a_ii**2-r_oii**2)-W_ssteel[8])
        Q_b             =  w_disc_r*0.5/inputs['R_sh']*(a_ii**2-r_oii**2)
        
        y_aiir          =  M_rb*a_ii**2/W_ssteel[0]*W_ssteel[1]+Q_b*a_ii**3/W_ssteel[0]*W_ssteel[2]-w_disc_r*a_ii**4/W_ssteel[0]*W_ssteel[7]
        
        I               =  pi*0.25*(R**4-(inputs['R_sh'])**4)
        F_ecc           = inputs['q']*2*pi*inputs['K_rad']*inputs['r_g']**3
        M_ar             = F_ecc*L_r*0.5
               
        
        outputs['y_ar'] =abs(y_ai1r+y_ai2r+y_ai3r)+y_aiir+(R+inputs['h_yr']+inputs['h_sr'])*inputs['theta_sh']+M_ar*L_r**2*0/(2*E*I)
        
        outputs['y_allowable_r'] =L_r/100*inputs['y_allow_pcent']
        
        # Torsional deformation of rotor
        J_dr            =0.5*pi*((R+inputs['h_yr']+inputs['h_sr'])**4-inputs['R_sh']**4)
        
        J_cylr          =0.5*pi*((R+inputs['h_yr']+inputs['h_sr'])**4-R**4)
        
        outputs['twist_r']=180/pi*inputs['T_rated']/G*(inputs['t_r']/J_dr+(L_r-inputs['t_r'])/J_cylr)
        
        outputs['Structural_mass_rotor'] = inputs['rho_Fes']*pi*(((R+inputs['h_yr']+inputs['h_sr'])**2-(inputs['R_sh'])**2)*inputs['t_r']+\
                                           ((R+inputs['h_yr']+inputs['h_sr'])**2-(R+inputs['h_yr'])**2)*inputs['l_s'])
                                           
        outputs['TC1r']  = (R+(inputs['h_yr']+inputs['h_sr']))**2*L_r
        outputs['TC2r']  = inputs['T_rated']/(2*pi*inputs['Sigma'])
        
        if outputs['TC1r']>outputs['TC2r']:
            outputs['TC_test_r']=1
        else:
            outputs['TC_test_r']=0
        
        
        
        
          			
class PMSG_stator_inactive(ExplicitComponent):
    """ Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator. """
    
    def setup(self):
    
        self.add_input('R_no',0.0, units ='m', desc='Nose outer radius ')
    
        self.add_input('y_bd', units ='W', desc='Deflection of the bedplate')
        self.add_input('theta_bd', 0.0, units = 'm', desc='Slope at the bedplate')
        
        self.add_input('T_rated',0.0, units = 'N*m', desc='Rated torque ')
        self.add_input('r_g',0.0, units ='m', desc='air gap radius')
        self.add_input('g',0.0, units ='m', desc='air gap length')
        self.add_input('h_t',0.0, units ='m', desc='tooth height')
        self.add_input('l_s',0.0, units ='m', desc='core length')
        self.add_input('q',0.0, units ='N/m**2', desc='Normal stress')
        self.add_input('h_ys',0.0, units ='m', desc='Stator yoke height ')
        
        # structural design variables
        
        self.add_input('t_s',0.0, units ='m', desc='Stator disc thickness')
        self.add_input('h_ss',0.0, units ='m', desc='Stator yoke height ')
        self.add_input('K_rad', desc='Aspect ratio')
        
        self.add_input('u_allow_pcent',0.0,desc='Radial deflection as a percentage of air gap diameter')
        self.add_input('y_allow_pcent',0.0,desc='Axial deflection as a percentage of air gap diameter')
        self.add_input('z_allow_deg',0.0,units='deg',desc='Allowable torsional twist')
        
        
        # Material properties
        self.add_input('rho_Fes',0.0, units='kg/(m**3)',desc='Structural Steel density')
        self.add_input('rho_Fe',0.0, units='kg/(m**3)', desc='Electrical Steel density ')
        self.add_input('Mass_tooth_stator',0.0,units ='kg', desc='Stator teeth mass ')
        self.add_input('Copper',0.0,units ='kg', desc='Copper mass ')

        
        self.add_output('Structural_mass_stator',0.0, units='kg', desc='Stator mass (kg)')
        self.add_output('u_as',0.0,units='m', desc='Radial deformation')
        self.add_output('y_as',0.0, units ='m', desc='Axial deformation')
        self.add_output('twist_s', 0.0, units ='deg', desc='Stator torsional twist')
        self.add_input('Sigma',0.0,units='N/m**2',desc='Shear stress')
               
        self.add_output('TC1s',0.0, desc='Torque constraint1-stator')
        self.add_output('TC2s',0.0, desc='Torque constraint2-stator')
        self.add_output('TC_test_s',0.0, desc='Torque constraint flag')
        
        self.add_output('u_allowable_s',0.0,units='m',desc='Allowable Radial deflection as a percentage of air gap diameter')
        self.add_output('y_allowable_s',0.0,units='m',desc='Allowable Axial deflection as a percentage of air gap diameter')
        
        
    def compute(self, inputs, outputs):
        
        #Assign values to universal constants
        
       
        # Radial deformation of Stator
        L_s             = inputs['l_s']+inputs['t_s']
        R_s             = inputs['r_g']-inputs['g']-inputs['h_t']-inputs['h_ys']-inputs['h_ss']
        constants_x_0   = shell_constant(R_s,inputs['t_s'],L_s,0,v)
        constants_x_L   = shell_constant(R_s,inputs['t_s'],L_s,L_s,v)
        f_d_denom1      = R_s/(E*((R_s)**2-(inputs['R_no'])**2))*((1-v)*R_s**2+(1+v)*(inputs['R_no'])**2)
        f_d_denom2      = inputs['t_s']/(2*constants_x_0[0]*(constants_x_0[1])**3)*(constants_x_0[2]/(2*constants_x_0[3])*constants_x_0[4]-constants_x_0[5]/constants_x_0[3]*constants_x_0[6]-0.5*constants_x_0[7])
        f               = inputs['q']*(R_s)**2*inputs['t_s']/(E*(inputs['h_ys']+inputs['h_ss'])*(f_d_denom1+f_d_denom2))
        outputs['u_as'] = (inputs['q']*(R_s)**2)/(E*(inputs['h_ys']+inputs['h_ss']))-f*0/(constants_x_L[0]*(constants_x_L[1])**3)*((constants_x_L[2]/(2*constants_x_L[3])*constants_x_L[4] -constants_x_L[5]/constants_x_L[3]*constants_x_L[6]-1/2*constants_x_L[7]))+inputs['y_bd']
        
        outputs['u_as'] = abs(outputs['u_as'] + inputs['y_bd'])
        
        outputs['u_allowable_s'] =2*inputs['r_g']/1000*inputs['u_allow_pcent']/100
        
        ###################################################### Electromagnetic design#############################################
        
        # axial deformation of stator
        W_back_iron     =  plate_constant(R_s+inputs['h_ss']+inputs['h_ys']+inputs['h_t'],inputs['R_no'],v,0.5*inputs['h_ys']+inputs['h_ss']+R_s,inputs['t_s'])
        W_ssteel        =  plate_constant(R_s+inputs['h_ss']+inputs['h_ys']+inputs['h_t'],inputs['R_no'],v,R_s+inputs['h_ss']*0.5,inputs['t_s'])
        W_active        =  plate_constant(R_s+inputs['h_ss']+inputs['h_ys']+inputs['h_t'],inputs['R_no'],v,R_s+inputs['h_ss']+inputs['h_ys']+inputs['h_t']*0.5,inputs['t_s'])
        
        W_is            =  inputs['rho_Fe']*g1*sin(phi)*(L_s-inputs['t_s'])*inputs['h_ys']
        y_ai1s           = -W_is*(0.5*inputs['h_ys']+R_s)**4/(inputs['R_no']*W_back_iron[0])*(W_back_iron[1]*W_back_iron[4]/W_back_iron[3]-W_back_iron[2])
                
        W_ss            =  inputs['rho_Fes']*g1*sin(phi)*(L_s-inputs['t_s'])*inputs['h_ss']
        y_ai2s           = -W_ss*(inputs['h_ss']*0.5+inputs['h_ys']+R_s)**4/(inputs['R_no']*W_ssteel[0])*(W_ssteel[1]*W_ssteel[4]/W_ssteel[3]-W_ssteel[2])
        W_cu            =  sin(phi)*inputs['Mass_tooth_stator']/(2*pi*(R_s+inputs['h_ss']+inputs['h_ys']+inputs['h_t']*0.5))
        y_ai3s           = -W_cu*(R_s+inputs['h_ss']+inputs['h_ys']+inputs['h_t']*0.5)**4/(inputs['R_no']*W_active[0])*(W_active[1]*W_active[4]/W_active[3]-W_active[2])
        
        w_disc_s        = inputs['rho_Fes']*g1*sin(phi)*inputs['t_s']
        
        a_ii            = R_s+inputs['h_ss']+inputs['h_ys']+inputs['h_t']
        r_oii           = inputs['R_no']
        M_rb            = -w_disc_s*a_ii**2/W_ssteel[5]*(W_ssteel[6]*0.5/(a_ii*inputs['R_no'])*(a_ii**2-r_oii**2)-W_ssteel[8])
        Q_b             =  w_disc_s*0.5/inputs['R_no']*(a_ii**2-r_oii**2)
        
        y_aiis          =  M_rb*a_ii**2/W_ssteel[0]*W_ssteel[1]+Q_b*a_ii**3/W_ssteel[0]*W_ssteel[2]-w_disc_s*a_ii**4/W_ssteel[0]*W_ssteel[7]
        
        I               =  pi*0.25*(R_s**4-(inputs['R_no'])**4)
        F_ecc           = inputs['q']*2*pi*inputs['K_rad']*inputs['r_g']**2
        M_as             = F_ecc*L_s*0.5
        
        outputs['y_as'] =abs(y_ai1s+y_ai2s+y_ai3s+y_aiis+(R_s+inputs['h_ys']+inputs['h_ss']+inputs['h_t'])*inputs['theta_bd'])+M_as*L_s**2*0/(2*E*I)
        
        
        outputs['y_allowable_s'] =L_s*inputs['y_allow_pcent']/100
        
        # Torsional deformation of stator
        J_ds            =0.5*pi*((R_s+inputs['h_ys']+inputs['h_ss']+inputs['h_t'])**4-inputs['R_no']**4)
        
        J_cyls          =0.5*pi*((R_s+inputs['h_ys']+inputs['h_ss']+inputs['h_t'])**4-R_s**4)
        
        outputs['twist_s']=180.0/pi*inputs['T_rated']/G*(inputs['t_s']/J_ds+(L_s-inputs['t_s'])/J_cyls)
        
        outputs['Structural_mass_stator'] = inputs['rho_Fes']*(pi*((R_s+inputs['h_ys']+inputs['h_ss']+inputs['h_t'])**2-(inputs['R_no'])**2)*inputs['t_s']+\
                                            pi*((R_s+inputs['h_ss'])**2-R_s**2)*inputs['l_s'])
   
        outputs['TC1s']  = (R_s+inputs['h_ys']+inputs['h_ss']+inputs['h_t'])**2*L_s
        outputs['TC2s']  = inputs['T_rated']/(2*pi*inputs['Sigma'])
        
        if outputs['TC1s']>outputs['TC2s']:
            outputs['TC_test_s']=1
        else:
            outputs['TC_test_s']=0
        


class PMSG_Constraints(ExplicitComponent):
    
    
    def setup(self):
        
        self.add_output('con_uar', val=0.0)
        self.add_output('con_yar', val=0.0)
        self.add_output('con_uas', val=0.0)
        self.add_output('con_yas', val=0.0)
        

        self.add_input('u_ar',0.0,units='m', desc='Radial deformation')
        self.add_input('y_ar',0.0, units ='m', desc='Axial deformation')
          
       
        self.add_input('u_allowable_r',0.0,units='m',desc='Allowable Radial deflection')
        self.add_input('y_allowable_r',0.0,units='m',desc='Allowable Radial deflection')
        self.add_input('z_allowable_r',0.0,units='m',desc='Allowable Circumferential deflection')
        
        self.add_input('u_as',0.0,units='m', desc='Radial deformation')
        self.add_input('y_as',0.0, units ='m', desc='Axial deformation')
        self.add_input('u_allowable_s',0.0,units='m',desc='Allowable Radial deflection as a percentage of air gap diameter')
        self.add_input('y_allowable_s',0.0,units='m',desc='Allowable Axial deflection as a percentage of air gap diameter')
        self.add_input('z_allowable_s',0.0,units='m',desc='Allowable Circumferential deflection')
        
        
    def compute(self, inputs, outputs):
    
        outputs['con_uar'] = 1e3*(inputs['u_allowable_r'] - inputs['u_ar'])
        outputs['con_yar'] = 1e3*(inputs['y_allowable_r'] - inputs['y_ar'])
        outputs['con_uas'] = 1e3*(inputs['u_allowable_s'] - inputs['u_as'])
        outputs['con_yas'] = 1e3*(inputs['y_allowable_s'] - inputs['y_as'])
        
        

####################################################Cost Analysis#######################################################################
class PMSG_Cost(ExplicitComponent):
    """ Provides a material cost estimate for a PMSG _arms generator. Manufacturing costs are excluded"""
    # Inputs
    # Specific cost of material by type
    
    def setup(self):
        
        self.add_input('C_Cu',0.0, units='USD/kg', desc='Specific cost of copper')
        self.add_input('C_Fe',0.0, units='USD/kg', desc='Specific cost of magnetic steel/iron')
        self.add_input('C_Fes',0.0, units='USD/kg', desc='Specific cost of structural steel')
        self.add_input('C_PM',0.0, units='USD/kg' , desc='Specific cost of Magnet')
        
        # Mass of each material type
        self.add_input('Copper',0.0, units ='kg',desc='Copper mass')
        self.add_input('Iron',0.0, units = 'kg', desc='Iron mass')
        self.add_input('mass_PM', 0.0, units ='kg',desc='Magnet mass')
        
        self.add_input('Structural_mass',0.0, units='kg', desc='Structural mass')
        # Outputs
        self.add_output('Costs',0.0,units ='USD', desc='Total cost')
        
    def compute(self, inputs, outputs):
    
        # Material cost as a function of material mass and specific cost of material
        
        K_gen=inputs['Copper']*inputs['C_Cu']+inputs['Iron']*inputs['C_Fe']+inputs['C_PM']*inputs['mass_PM']
        
        Cost_str=inputs['C_Fes']*inputs['Structural_mass']
        
        outputs['Costs']=K_gen +Cost_str
		
  
####################################################OPTIMISATION SET_UP ###############################################################

class PMSG_Outer_rotor_Opt(Group):
    
    def setup(self):
        
        Outer_rotorIndeps = IndepVarComp()
        Outer_rotorIndeps.add_discrete_output('Eta_target',0.0,desc='Target drivetrain efficiency')
        Outer_rotorIndeps.add_output('P_rated', 0.0,units='W',desc='Rated Power')
        Outer_rotorIndeps.add_output('T_rated', 0.0,units='N*m',desc='Torque')
        Outer_rotorIndeps.add_output('N_nom', 0.0,units='rpm',desc='rated speed')
        Outer_rotorIndeps.add_output('r_g', 0.0,units='m',desc='Air gap radius')
        Outer_rotorIndeps.add_output('l_s', 0.0,units='m',desc='Core length')
        Outer_rotorIndeps.add_output('h_s', 0.0,units='m',desc='Slot height')
        Outer_rotorIndeps.add_output('p', 0.0,desc='Pole pairs')
        Outer_rotorIndeps.add_output('h_m', 0.0,units='m',desc='Magnet height' )
        Outer_rotorIndeps.add_output('h_yr', 0.0,units='m',desc='Rotor yoke height'  )
        Outer_rotorIndeps.add_output('h_ys', 0.0,units='m',desc='Stator yoke height')
        Outer_rotorIndeps.add_output('B_tmax',0.0,desc='Teeth flux density')
        Outer_rotorIndeps.add_output('t_r', 0.0,units='m',desc='Rotor disc thickness')
        Outer_rotorIndeps.add_output('t_s', 0.0,units='m',desc='Stator disc thickness' )
        Outer_rotorIndeps.add_output('h_ss',0.0,units='m',desc='Stator rim thickness')
        Outer_rotorIndeps.add_output('h_sr', 0.0,units='m',desc='Rotor rim thickness')    
        

        Outer_rotorIndeps.add_output('rho_Fe', 0.0,units='kg/m**3',desc='Electrical steel density')
        Outer_rotorIndeps.add_output('rho_Fes', 0.0,units='kg/m**3',desc='Structural steel density')
        

        
        Outer_rotorIndeps.add_output('u_allow_pcent', 0.0,desc='% radial deflection')
        Outer_rotorIndeps.add_output('y_allow_pcent', 0.0,desc='% axial deflection')
        Outer_rotorIndeps.add_output('Sigma', 0.0,units='N/m**2',desc='shear stress')
        
        Outer_rotorIndeps.add_output('z_allow_deg',0.0,units='deg',desc='Allowable torsional twist')
                
        self.add_subsystem('Outer_rotorIndeps',Outer_rotorIndeps, promotes =['*'])
        self.add_subsystem('PMSG_active',PMSG_active(), promotes =['*'])
        self.add_subsystem('PMSG_rotor_inactive',PMSG_rotor_inactive(), promotes =['*'])
        self.add_subsystem('PMSG_stator_inactive',PMSG_stator_inactive(), promotes =['*'])
        self.add_subsystem('PMSG_Cost',PMSG_Cost(), promotes =['*'])
        self.add_subsystem('PMSG_Constraints',PMSG_Constraints(), promotes =['*'])
        
       
		
if __name__ == '__main__':

    prob = Problem()
    
 
    prob.model = PMSG_Outer_rotor_Opt()
    
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'CONMIN' #'COBYLA'
    prob.driver.opt_settings['IPRINT'] = 4
    prob.driver.opt_settings['ITRM'] = 3
    prob.driver.opt_settings['ITMAX'] = 1000
    # prob.driver.opt_settings['DELFUN'] = 1e-3
    # prob.driver.opt_settings['DABFUN'] = 1e-3
    prob.driver.opt_settings['IFILE'] = 'CONMIN_PMSG_disc.out'
    # prob.root.deriv_options['type']='fd'
    
    
   
   
    prob.model.add_constraint('B_symax',lower=0.0,upper=2.0)									#1
    prob.model.add_constraint('B_rymax',lower=0.0,upper=2.0)									#2
    prob.model.add_constraint('b_t',    lower=0.01)				    #3
    prob.model.add_constraint('B_g', lower=0.7,upper=1.3)                           #4
    prob.model.add_constraint('E_p',    lower=500, upper=10000)						#5
    prob.model.add_constraint('A_Cuscalc',lower=5.0,upper=240) 						#8
    prob.model.add_constraint('K_rad',    lower=0.15,upper=0.3)						#10
    prob.model.add_constraint('Slot_aspect_ratio',lower=4.0, upper=10.0)			#11
    prob.model.add_constraint('gen_eff',lower=95.5)						            #14
    prob.model.add_constraint('A_1',upper=95000.0)					#16
    prob.model.add_constraint('T_e', lower= 21.03e6,upper=21.1e6)
    prob.model.add_constraint('J_actual',lower=3,upper=6) 						#8
    
    
    # structural constraints
    prob.model.add_constraint('con_uar',lower = 1e-2)  #1e-2           #17
    prob.model.add_constraint('con_yar', lower = 1e-2)                             #18
    prob.model.add_constraint('con_uas', lower = 1e-2)                             #20
    prob.model.add_constraint('con_yas',lower = 1e-2)                             #21
   
    
    prob.model.add_objective('Mass',scaler=1e-4)
    
    prob.model.add_design_var('r_g', lower=3.0, upper=9 ) 
    prob.model.add_design_var('l_s', lower=1.5, upper=2.5 )  
    prob.model.add_design_var('h_s', lower=0.1, upper=1.0 )  
    prob.model.add_design_var('p', lower=50.0, upper=100)
    prob.model.add_design_var('h_m', lower=0.01, upper=0.2 )  
    prob.model.add_design_var('h_yr', lower=0.035, upper=0.22 )
    prob.model.add_design_var('h_ys', lower=0.035, upper=0.22 )
    prob.model.add_design_var('B_tmax', lower=1, upper=2.0 ) 
    prob.model.add_design_var('t_r', lower=0.05, upper=0.3 ) 
    prob.model.add_design_var('t_s', lower=0.05, upper=0.3 )  
    prob.model.add_design_var('h_ss', lower=0.04, upper=0.2)
    prob.model.add_design_var('h_sr', lower=0.04, upper=0.2)
    
    
    prob.setup()
    
    
        # --- Design Variables ---

    
	#Initial design variables for a PMSG designed for a 15MW turbine
    
    prob['P_rated']     =   15000000.0
    
    prob['T_rated']     =   21.03065891799495e6         #rev 1 9.94718e6
    
    
    
    prob['N_nom']       =   7.5598598  #8.68                # rpm 9.6
    
    prob['r_g']         =   5.15     # rev 1 6.8
   
    prob['l_s']         =   2.25 	# rev 2.1
    
    prob['h_s']         =  0.4 # rev 1 0.3
    
    prob['p']           =   100.0 #100.0    # rev 1 160
    
    
    prob['h_m']         =   0.090   # rev 1 0.034
    
    prob['h_ys']        =   0.04 # rev 1 0.045
    
    prob['h_yr']        =   0.04 # rev 1 0.045
    
   
    prob['b']   =   2.
    
    prob['c']   =5.0
    prob['B_tmax']      =   1.9 #
#	
    # Specific costs
    prob['C_Cu']        =   4.786
    prob['C_Fe']    	=   0.556
    prob['C_Fes']       =   0.50139
    prob['C_PM']        =   95.0
    
    #Material properties
    
    prob['rho_Fe']      =   7700.0                 #Steel density
    prob['rho_Fes']     =   7850.0                 #Steel density
    prob['rho_Copper']  =   8900.0                  # Kg/m3 copper density
    prob['rho_PM']      =   7400.0                  # magnet density
    prob['rho_Cu']      =   1.8*10**(-8)*1.4			# Copper resisitivty
    
    #Support structure parameters
    prob['R_no']        = 1.1		# Nose outer radius
    prob['R_sh']        = 1.625 			# Shaft outer radius =(2+0.25*2+0.3*2)*0.5
    
    prob['t_r']         =   0.06 			# Rotor disc thickness
    prob['h_sr']        =   0.04           # Rotor cylinder thickness

    prob['t_s']         =   0.06 			# Stator disc thickness
    prob['h_ss']        =   0.04           # Stator cylinder thickness
    
    prob['y_sh']         =   0.0005*0			# Shaft deflection
    prob['theta_sh']     =   0.00026*0      # Slope at shaft end

    prob['y_bd']         =   0.0005*0# deflection at bedplate
    prob['theta_bd']     =  0.00026*0   # Slope at bedplate end
    
    prob['u_allow_pcent']=  8.5       # % radial deflection
    prob['y_allow_pcent']=  1.0       # % axial deflection
    
    prob['z_allow_deg']  =  0.05       # torsional twist
    
    prob['Sigma']        =  60.0e3       # Shear stress
      

        
    prob.model.approx_totals(method='fd')    
    
    
    prob.run_driver()
    
    #prob.model.list_outputs(values = True, hierarchical=True)
    
    raw_data = {'Parameters': ['Rating','Air gap diameter','Overall Outer diameter', 'Stator length', 'l/d ratio',\
                'Slot_aspect_ratio','Pole pitch','Slot pitch', 'Stator slot height','Stator slotwidth','Stator tooth width','Stator tooth height',\
                'Stator yoke height', 'Rotor yoke height', 'Magnet height', 'Magnet width', 'Peak air gap flux density fundamental',\
                'Peak stator yoke flux density','Peak rotor yoke flux density','Flux density above magnet','Maximum Stator flux density',\
                'Maximum tooth flux density','Pole pairs', 'Generator output frequency', 'Generator output phase voltage',\
                'Generator Output phase current', 'Stator resistance','Synchronous inductance', 'Stator slots','Stator turns',\
                'Conductor cross-section','Stator Current density ','Specific current loading','Generator Efficiency ','Iron mass',\
                'Magnet mass','Copper mass','Structural Mass','**************************','Rotor disc thickness','Rotor rim thickness',\
                'Stator disc thickness','Stator rim thickness','Rotor radial deflection','Rotor axial deflection','Rotor twist','Torsional Constraint',\
                'Stator radial deflection','Stator axial deflection','Stator twist','Torsional Constraint','**************************','Total Mass','Total Material Cost'],
    'Values': [prob['P_rated']/1000000,2*prob['r_g'],prob['R_out']*2,prob['l_s'],prob['K_rad'],prob['Slot_aspect_ratio'],\
                prob['tau_p']*1000,prob['tau_s']*1000,prob['h_s']*1000,prob['b_s']*1000,prob['b_t']*1000,prob['h_t']*1000,prob['h_ys']*1000,\
                prob['h_yr']*1000,prob['h_m']*1000,prob['b_m']*1000,prob['B_g'],prob['B_symax'],prob['B_rymax'],prob['B_pm1'],\
                prob['B_smax'],prob['B_tmax'],prob['p1'],prob['f'],prob['E_p'],prob['I_s'],prob['R_s'],prob['L_s'],prob['S'],\
                prob['N_s'],prob['A_Cuscalc'],prob['J_actual'],prob['A_1']/1000,prob['gen_eff'],prob['Iron']/1000,\
                prob['mass_PM']/1000,prob['Copper']/1000,prob['Structural_mass']/1000,'************************',prob['t_r']*1000,\
                prob['h_sr']*1000,prob['t_s']*1000,prob['h_ss']*1000,prob['u_ar']*1000,prob['y_ar']*1000,prob['twist_r'],prob['TC_test_r']\
                ,prob['u_as']*1000,prob['y_as']*1000,prob['twist_s'],prob['TC_test_s'],'**************************',prob['Mass']/1000,prob['Costs']/1000],
    'Limit': ['','','','','(0.15-0.3)','(4-10)','','','','','','','','','','','<2','<2','<2',prob['B_g'],'<2','<2','','','500<E_p<10000','','','','','','',\
    '3-6','85','>=95.4','','','','','************************','','','','',prob['u_allowable_r']*1000,prob['y_allowable_r']*1000,\
    prob['z_allow_deg'],'',prob['u_allowable_s']*1000,prob['y_allowable_s']*1000,prob['z_allow_deg'],'','**************************','',''],
    'Units':['MW','m','m','m','','','mm','mm','mm','mm','mm','mm','mm','mm','mm','mm','T','T','T','T','T','T','-','Hz','V','A','ohm/phase','p.u','slots','turns','mm^2','A/mm^2',\
            'kA/m','%','tons','tons','tons','tons','************','mm','mm','mm','mm','mm','mm','deg','','mm','mm','deg','','************','tons','k$']}
    
    df=pandas.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
    
    print (df)
    
    df.to_excel('Optimized_PMSG_'+str(prob['P_rated'][0]/1e6)+'_MW.xlsx')
	
	
