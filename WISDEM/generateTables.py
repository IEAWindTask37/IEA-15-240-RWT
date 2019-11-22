import os
try:
    import ruamel.yaml as yaml
except:
    try:
        import ruamel_yaml as yaml
    except:
        raise ImportError('No YAML package found')
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

class RWT_Tabular(object):
    def __init__(self, finput, towDF=None, rotDF=None):
        
        # Read ontology file into dictionary-like data structure
        f = open(finput, 'r')
        self.yaml = yaml.safe_load( f )
        f.close()
        
        # Store output file name
        froot, _ = os.path.splitext( finput )
        self.fout = froot + '_tabular.xlsx'

        # If provided, store tower and rotor data
        self.towDF = towDF
        self.rotDF = rotDF

        # Initialize workbook object
        self.wb   = Workbook()

        
    def write_all(self):
        self.write_overview()
        self.write_blade_outer()
        self.write_airfoils()
        self.write_blade_inner()
        self.write_tower_monopile()
        self.write_materials()
        self.write_rotor_performance()
        self.cleanup()

    def write_overview(self):
        # Sheet name
        ws = self.wb.create_sheet(title = 'Overview')
        ws['A1'] = 'Parameter'
        ws['B1'] = 'IEA 15 MW Reference Wind Turbine'

        irow = 2
        ws.cell(row=irow, column=1, value='Power rating [MW]')
        ws.cell(row=irow, column=2, value=15.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Turbine class')
        ws.cell(row=irow, column=2, value='IEC Class 1B')
        irow += 1
        
        ws.cell(row=irow, column=1, value='Specific rating [W/m^2]')
        ws.cell(row=irow, column=2, value=15e6/np.pi/120.0**2.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Rotor orientation')
        ws.cell(row=irow, column=2, value='Upwind')
        irow += 1
        
        ws.cell(row=irow, column=1, value='Number of blades')
        ws.cell(row=irow, column=2, value=3)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Control')
        ws.cell(row=irow, column=2, value='Variable speed, Collective pitch')
        irow += 1
        
        ws.cell(row=irow, column=1, value='Cut-in wind speed [m/s]')
        ws.cell(row=irow, column=2, value=3.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Rated wind speed [m/s]')
        ws.cell(row=irow, column=2, value=10.56)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Cut-out wind speed [m/s]')
        ws.cell(row=irow, column=2, value=25.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Rotor diameter [m]')
        ws.cell(row=irow, column=2, value=240.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Airfoil series')
        ws.cell(row=irow, column=2, value='FFA-W3')
        irow += 1
        
        ws.cell(row=irow, column=1, value='Hub height [m]')
        ws.cell(row=irow, column=2, value=150.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Hub diameter [m]')
        ws.cell(row=irow, column=2, value=6.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Hub Overhang [m]')
        ws.cell(row=irow, column=2, value='TODO??')
        irow += 1
        
        ws.cell(row=irow, column=1, value='Drive train')
        ws.cell(row=irow, column=2, value='Low speed, Direct drive')
        irow += 1

        ws.cell(row=irow, column=1, value='Design tip speed ratio')
        ws.cell(row=irow, column=2, value=9.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Minimum rotor speed [rpm]')
        ws.cell(row=irow, column=2, value=4.6)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Maximum rotor speed [rpm]')
        ws.cell(row=irow, column=2, value=7.6)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Maximum tip speed [m/s]')
        ws.cell(row=irow, column=2, value=95.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Shaft tilt angle [deg]')
        ws.cell(row=irow, column=2, value=6.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Rotor pre-cone angle [deg]')
        ws.cell(row=irow, column=2, value=-4.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Tower top to nacelle (yaw bearing height) [m]')
        ws.cell(row=irow, column=2, value=0.385)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Tower top to shaft distnace [m]')
        ws.cell(row=irow, column=2, value=2.927)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Overhang [m]')
        ws.cell(row=irow, column=2, value=-10.454)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Generator efficiency [%]')
        ws.cell(row=irow, column=2, value=96.55)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Blade pre-bend [m]')
        ws.cell(row=irow, column=2, value=4.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Blade mass [t]')
        ws.cell(row=irow, column=2, value=68.0)
        irow += 1

        ws.cell(row=irow, column=1, value='Hub mass [t]')
        ws.cell(row=irow, column=2, value=188.0)
        irow += 1

        ws.cell(row=irow, column=1, value='Generator mass [t]')
        ws.cell(row=irow, column=2, value=371.950)
        irow += 1

        ws.cell(row=irow, column=1, value='Nose mass [t]')
        ws.cell(row=irow, column=2, value=12.947)
        irow += 1

        ws.cell(row=irow, column=1, value='Bedplate mass [t]')
        ws.cell(row=irow, column=2, value=49.999)
        irow += 1

        ws.cell(row=irow, column=1, value='Shaft mass [t]')
        ws.cell(row=irow, column=2, value=19.504)
        irow += 1

        ws.cell(row=irow, column=1, value='Shaft bearing mass [t]')
        ws.cell(row=irow, column=2, value=4.699)
        irow += 1

        ws.cell(row=irow, column=1, value='Flange mass [t]')
        ws.cell(row=irow, column=2, value=1.739)
        irow += 1

        ws.cell(row=irow, column=1, value='Yaw system mass [t]')
        ws.cell(row=irow, column=2, value=100.0)
        irow += 1

        ws.cell(row=irow, column=1, value='Other nacelle mass (electronics, thermal system, etc) [t]')
        ws.cell(row=irow, column=2, value=50.0)
        irow += 1

        ws.cell(row=irow, column=1, value='Nacelle mass [t]')
        ws.cell(row=irow, column=2, value=615.537)
        irow += 1
        
        ws.cell(row=irow, column=1, value='RNA mass [t]')
        ws.cell(row=irow, column=2, value=1007.537)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Nacellem center of mass from tower top [m]')
        ws.cell(row=irow, column=2, value='[-5.019, 0.0, 3.161]')
        irow += 1
        
        ws.cell(row=irow, column=1, value='Tower mass [t]')
        ws.cell(row=irow, column=2, value='TODO??')
        irow += 1
        
        ws.cell(row=irow, column=1, value='Tower base diameter [m]')
        ws.cell(row=irow, column=2, value=10.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Transition piece height [m]')
        ws.cell(row=irow, column=2, value=10.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Monopile embedment depth [m]')
        ws.cell(row=irow, column=2, value='TODO??')
        irow += 1
        
        ws.cell(row=irow, column=1, value='Monopile base diameter [m]')
        ws.cell(row=irow, column=2, value=10.0)
        irow += 1
        
        ws.cell(row=irow, column=1, value='Monopile mass [t]')
        ws.cell(row=irow, column=2, value='TODO??')
        irow += 1
        
        # Header row style formatting
        for cell in ws["1:1"]:
            cell.style = 'Headline 2'
 
 
        
    def write_airfoils(self):
        # Airfoil csv files
        for iaf in range(len(self.yaml['airfoils'])):
            # Sheet name will be airfoil name
            ws = self.wb.create_sheet(title = 'Airfoil '+self.yaml['airfoils'][iaf]['name'])

            # Write name and aerodynamic center
            ws['A1'] = 'name'
            ws['B1'] = self.yaml['airfoils'][iaf]['name']

            ws['A2'] = 'relative_thickness'
            ws['B2'] = self.yaml['airfoils'][iaf]['relative_thickness']

            ws['A3'] = 'aerodynamic_center'
            ws['B3'] = self.yaml['airfoils'][iaf]['aerodynamic_center']

            # Write airfoil xy-coordinates
            ws['A5'] = 'x'
            ws['B5'] = 'y'
            nxy      = len(self.yaml['airfoils'][iaf]['coordinates']['x'])
            for k in range(nxy):
                ws.cell(row=k+6, column=1, value=self.yaml['airfoils'][iaf]['coordinates']['x'][k])
                ws.cell(row=k+6, column=2, value=self.yaml['airfoils'][iaf]['coordinates']['y'][k])

            # Write airfoil polars
            row_start = 6+nxy+2
            for ipol in range(len(self.yaml['airfoils'][iaf]['polars'])):
                # Reynolds number label
                ws.cell(row=row_start, column=1, value='Reynolds')
                ws.cell(row=row_start, column=2, value=self.yaml['airfoils'][iaf]['polars'][ipol]['re'])

                # Create data array in pandas
                polmat = np.c_[self.yaml['airfoils'][iaf]['polars'][ipol]['c_l']['grid'],
                               self.yaml['airfoils'][iaf]['polars'][ipol]['c_l']['values'],
                               self.yaml['airfoils'][iaf]['polars'][ipol]['c_d']['values'],
                               self.yaml['airfoils'][iaf]['polars'][ipol]['c_m']['values']]
                polDF = pd.DataFrame(polmat, columns=['alpha [rad]',
                                                      'c_l',
                                                      'c_d',
                                                      'c_m'])

                # Append to worksheet
                for r in dataframe_to_rows(polDF, index=False, header=True):
                    ws.append(r)

                # Prep for next polar
                row_start += polmat.shape[0]+1

                
    def write_blade_outer(self):
        # Sheet name
        ws = self.wb.create_sheet(title = 'Blade Geometry')

        # Write airfoil xy-coordinates
        ws['A1'] = 'Spanwise position [r/R]'
        ws['B1'] = 'Airfoil name'
        npts = len(self.yaml['components']['blade']['outer_shape_bem']['airfoil_position']['grid'])
        for k in range(npts):
            ws.cell(row=k+2, column=1, value=self.yaml['components']['blade']['outer_shape_bem']['airfoil_position']['grid'][k])
            ws.cell(row=k+2, column=2, value=self.yaml['components']['blade']['outer_shape_bem']['airfoil_position']['labels'][k])

        ws.cell(row=npts+3, column=1, value='Profile')
        # Create blade geometry array in pandas
        geommat = np.c_[self.yaml['components']['blade']['outer_shape_bem']['chord']['grid'],
                        self.yaml['components']['blade']['outer_shape_bem']['chord']['values'],
                        self.yaml['components']['blade']['outer_shape_bem']['twist']['values'],
                        self.yaml['components']['blade']['outer_shape_bem']['pitch_axis']['values'],
                        self.yaml['components']['blade']['outer_shape_bem']['reference_axis']['z']['values'],
                        self.yaml['components']['blade']['outer_shape_bem']['reference_axis']['x']['values'],
                        self.yaml['components']['blade']['outer_shape_bem']['reference_axis']['y']['values']]
        geomDF = pd.DataFrame(geommat, columns=['Spanwise position [r/R]',
                                              'Chord [m]',
                                              'Twist [rad]',
                                              'Pitch axis [x/c]',
                                              'Span [m]',
                                              'Prebend [m]',
                                              'Sweep [m]'])
        
        # Set convention that positive prebend means upward tilt to give tower strike margin
        geomDF['Prebend [m]'] *= -1.0
        
        # Append to worksheet
        for r in dataframe_to_rows(geomDF, index=False, header=True):
            ws.append(r)

        # Header row style formatting
        #for cell in ws["1:1"]:
        #    cell.style = 'Headline 2'
    
    def write_blade_inner(self):
        pass
    
    def write_tower_monopile(self):
        if not self.towDF is None:
            ws = self.wb.create_sheet(title = 'Tower Properties')
            for r in dataframe_to_rows(self.towDF, index=False, header=True):
                ws.append(r)
        
            # Header row style formatting
            for cell in ws["1:1"]:
                cell.style = 'Headline 2'
            
    
    def write_materials(self):
        # Sheet name
        ws = self.wb.create_sheet(title = 'Material Properties')

        # Initialize row counter
        irow = 1

        # Loop over all materials
        for k in range(len(self.yaml['materials'])):
            # Make sure material name is printer first
            ws.cell(row=irow, column=1, value='Name')
            ws.cell(row=irow, column=2, value=self.yaml['materials'][k]['name'])
            irow += 1

            # Loop over all other material properties and spit them out
            for key in self.yaml['materials'][k].keys():
                if key == 'name': continue

                # Catch if property value is a list
                myval = self.yaml['materials'][k][key]
                if type(myval) == type([]):
                    myval = str(myval)

                # Add units to keyname
                keystr = key
                if key in ['E','G','S','Xt','Xc']:
                    keystr += ' [Pa]'
                elif key in ['rho']:
                    keystr += ' [kg/m^3]'

                # Write to worksheet
                ws.cell(row=irow, column=1, value=keystr)
                ws.cell(row=irow, column=2, value=myval)
                irow += 1

            # Be sure to skip a line to the next material
            irow += 1
            
    
    def write_rotor_performance(self):
        if not self.rotDF is None:
            ws = self.wb.create_sheet(title = 'Rotor Performance')
            for r in dataframe_to_rows(self.rotDF, index=False, header=True):
                ws.append(r)
        
            # Header row style formatting
            for cell in ws["1:1"]:
                cell.style = 'Headline 2'

                
    def cleanup(self):
        # Remove empty sheet
        self.wb.active = 0
        ws = self.wb.active
        self.wb.remove(ws)
        
        # Save workbook to file
        self.wb.save(filename=self.fout)

        
if __name__ == '__main__':
    
    # Ontology file as input
    fontology = 'IEA-15-240-RWT.yaml'
    
    myobj = RWT_Tabular(fontology, fxlsx)
    myobj.write_all()
