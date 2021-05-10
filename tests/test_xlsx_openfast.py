import weio
import os
import pandas as pd
import unittest

FROOT = os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) )

class TestConsistency(unittest.TestCase):
    def test_nacelle_mass(self):
        # ElastoDyn vals
        ED_mono = weio.read( os.path.join(FROOT, 'OpenFAST', 'IEA-15-240-RWT-Monopile', 'IEA-15-240-RWT-Monopile_ElastoDyn.dat') )
        ED_semi = weio.read( os.path.join(FROOT, 'OpenFAST', 'IEA-15-240-RWT-UMaineSemi', 'IEA-15-240-RWT-UMaineSemi_ElastoDyn.dat') )

        # Excel tabular data
        tabdata = pd.read_excel( os.path.join(FROOT, 'Documentation', 'IEA-15-240-RWT_tabular.xlsx'),
                                 sheet_name='Nacelle Mass Properties', header=0, index_col=0,
                                 engine='openpyxl')
        
        # Check for consistency in nacelle mass props
        for ED in [ED_mono, ED_semi]:
            self.assertAlmostEqual(ED['HubMass'], tabdata.loc['Hub_System','Mass'], 0)
            self.assertAlmostEqual(ED['HubIner'], tabdata.loc['Hub_System','MoI_CoM_xx'], 0)
            self.assertAlmostEqual(ED['GenIner'], tabdata.loc['generator','MoI_CoM_xx'], 0)
            self.assertAlmostEqual(ED['NacMass'], tabdata.loc['Above_yaw','Mass'], 0)
            self.assertAlmostEqual(ED['NacCMxn'], tabdata.loc['Above_yaw','CoM_TT_x'], 0)
            self.assertAlmostEqual(ED['NacCMyn'], 0.0, 0)
            self.assertAlmostEqual(ED['NacCMzn'], tabdata.loc['Above_yaw','CoM_TT_z'], 0)
            self.assertAlmostEqual(ED['NacYIner'], tabdata.loc['Above_yaw','MoI_TT_zz'], 0)
            self.assertAlmostEqual(ED['YawBrMass'], tabdata.loc['yaw','Mass'], 0)
        

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestConsistency))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
