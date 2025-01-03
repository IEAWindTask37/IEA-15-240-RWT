"""Check that the yaml, Excel, HAWC2, and OpenFAST tower models match.

Specifically:
    1. That the tower definition in the yaml and the Excel match
    2. That the distributed mass-per-unit-length and stiffnes in the Excel,
       HAWC2, recalculated-from-yaml, and OpenFAST all match.

TODO: Add check for monopile model when added in HAWC2.
"""
import numpy as np

import _test_functions as tstf
from _test_functions import FROOT
import unittest

class TestConsistency(unittest.TestCase):

    def test_tower_matches(self):
        """1. Design in yaml-Excel match, 2. distprops in all 4 match."""

        # paths
        yaml_path = FROOT / 'WT_Ontology/IEA-15-240-RWT.yaml'  # yaml file with data
        ed_path = FROOT / 'OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_ElastoDyn_tower.dat'  # elastodyn file
        h2_ons_path = FROOT / 'HAWC2/IEA-15-240-RWT-Onshore/data/IEA_15MW_RWT_Tower_st.dat'  # hawc2 onshore st file
        h2_mon_path = FROOT / 'HAWC2/IEA-15-240-RWT-Monopile/data/IEA_15MW_RWT_Tower_st.dat'  # hawc2 monopile st file
        ex_path = FROOT / 'Documentation/IEA-15-240-RWT_tabular.xlsx'  # pth to excel file

        # load yaml values, calculate distributed properties
        yamldata = tstf.load_yaml(yaml_path)
        twr_stn_yaml, out_diam_yaml, thick_yaml, E, G, rho, outfit = tstf.load_body_properties('tower', yamldata)
        mpl_yaml = tstf.calculate_mpl(out_diam_yaml, thick_yaml, rho, outfitting_factor=outfit)
        EI_yaml = E * tstf.calculate_mom_iner(out_diam_yaml, thick_yaml)

        # load excel values, update the first two stations
        ex_df = tstf.load_excel_tower(ex_path)
        ex_df = ex_df.iloc[1:, :]  # remove 15.000 element
        ex_df.iloc[0, 0] = 15.  # overwrite the 15.001 to 15.0
        twr_stn_ex, out_diam_ex, thick_ex, mpl_ex, EIx_ex, EIy_ex = ex_df.iloc[:, [0, 1, 2, 3, 6, 7]].to_numpy().T
        thick_ex *= 1e-3  # mm to m

        # load the elastodyn tower properties
        ed_st = tstf.load_elastodyn_distprop(ed_path)
        mpl_ed, EIx_ed, EIy_ed = ed_st[:, 1:4].T

        # load the hawc2 onshore tower properties
        h2_ons = tstf.load_hawc2_st(h2_ons_path)
        mpl_h2_ons = h2_ons[:, 1]
        EIx_h2_ons = h2_ons[:, 8]*h2_ons[:, 10]
        EIy_h2_ons = h2_ons[:, 8]*h2_ons[:, 11]

        # load the hawc2 monopile tower properties
        h2_mon = tstf.load_hawc2_st(h2_mon_path)
        mpl_h2_mon = h2_mon[:, 1]
        EIx_h2_mon = h2_mon[:, 8]*h2_mon[:, 10]
        EIy_h2_mon = h2_mon[:, 8]*h2_mon[:, 11]

        # compare tower design in yaml and excel
        for (arr_yaml, arr_ex) in zip([twr_stn_yaml, out_diam_yaml, thick_yaml],
                                    [twr_stn_ex, out_diam_ex, thick_ex]):
            np.testing.assert_allclose(arr_yaml, arr_ex)

        # compare mpl and bending stiffness for all 4
        yaml_vals = [mpl_yaml, EI_yaml, EI_yaml]
        ex_vals = [mpl_ex, EIx_ex, EIy_ex]
        ed_vals = [mpl_ed, EIx_ed, EIy_ed]
        h2_ons_vals = [mpl_h2_ons, EIx_h2_ons, EIy_h2_ons]
        h2_mon_vals = [mpl_h2_mon, EIx_h2_mon, EIy_h2_mon]
        for (arr_yaml, arr_ex, arr_ed, arr_h2_ons, arr_h2_mon) in zip(yaml_vals, ex_vals, ed_vals, h2_ons_vals, h2_mon_vals):
            np.testing.assert_allclose(arr_yaml, arr_ex, rtol=1e-3)  # yaml versus excel
            np.testing.assert_allclose(arr_yaml, arr_ed, rtol=1e-3)  # yaml versus elastodyn
            np.testing.assert_allclose(arr_yaml, arr_h2_ons, rtol=1e-3)  # yaml versus hawc2 onshore
            np.testing.assert_allclose(arr_yaml, arr_h2_mon, rtol=1e-3)  # yaml versus hawc2 monopile




if __name__ == "__main__":
    unittest.main()
    
