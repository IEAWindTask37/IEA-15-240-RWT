"""Check that the yaml, Excel, HAWC2, and OpenFAST monopile models match.

Specifically:
    1. That the monopile definition in the yaml and the Excel match
    2. That the distributed mass-per-unit-length and stiffnes in the Excel,
       HAWC2, recalculated-from-yaml, and OpenFAST all match.

TODO: Add routine to load SubDyn distributed material properties.
"""
import numpy as np

import _test_functions as tstf
from _test_functions import FROOT


def test_monopile_matches():
    """1. Design in yaml-Excel match, 2. distprops in yaml/excel/hawc/openfast match."""

    # paths
    yaml_path = FROOT / 'WT_Ontology/IEA-15-240-RWT.yaml'  # yaml file with data
    h2_embmon_path = FROOT / 'HAWC2/IEA-15-240-RWT-Monopile/data/IEA_15MW_RWT_Embeddedmonopile_st.dat'  # embedded monopile st file
    h2_mon_path = FROOT / 'HAWC2/IEA-15-240-RWT-Monopile/data/IEA_15MW_RWT_Monopile_st.dat'  # monopile st file
    ex_path = FROOT / 'Documentation/IEA-15-240-RWT_tabular.xlsx'  # pth to excel file
    sd_path = FROOT / 'OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_SubDyn.dat'  # subdyn file

    # load yaml values, calculate distributed properties
    yamldata = tstf.load_yaml(yaml_path)
    twr_stn_yaml, out_diam_yaml, thick_yaml, E, G, rho, outfit = tstf.load_body_properties('monopile', yamldata)
    mpl_yaml = tstf.calculate_mpl(out_diam_yaml, thick_yaml, rho, outfitting_factor=outfit)
    EI_yaml = E * tstf.calculate_mom_iner(out_diam_yaml, thick_yaml)

    # load excel values, manually add -29.999 station
    ex_df = tstf.load_excel_tower(ex_path, body='monopile')
    ex_df.loc[1.5] = ex_df.loc[1]
    ex_df.loc[1.5, 'Height [m]'] = -29.999
    ex_df = ex_df.sort_index().reset_index(drop=True)
    twr_stn_ex, out_diam_ex, thick_ex, mpl_ex, EIx_ex, EIy_ex = ex_df.iloc[:, [0, 1, 2, 3, 6, 7]].to_numpy().T
    thick_ex *= 1e-3  # mm to m

    # compare tower design in yaml and excel
    for (arr_yaml, arr_ex) in zip([twr_stn_yaml, out_diam_yaml, thick_yaml],
                                [twr_stn_ex, out_diam_ex, thick_ex]):
        np.testing.assert_allclose(arr_yaml, arr_ex)

    # # load the subdyn tower properties TODO: Add subdyn loading routine
    mpl_sd, EI_sd, D, t = tstf.load_subdyn_distprop(sd_path, outfit)

    # load the hawc2 embedded-monopile tower properties
    h2_emdmon = tstf.load_hawc2_st(h2_embmon_path)
    mpl_h2_embmon = h2_emdmon[:, 1]
    EIx_h2_embmon = h2_emdmon[:, 8]*h2_emdmon[:, 10]
    EIy_h2_embmon = h2_emdmon[:, 8]*h2_emdmon[:, 11]

    # load the hawc2 monopile tower properties
    h2_mon = tstf.load_hawc2_st(h2_mon_path)
    mpl_h2_mon = h2_mon[:, 1]
    EIx_h2_mon = h2_mon[:, 8]*h2_mon[:, 10]
    EIy_h2_mon = h2_mon[:, 8]*h2_mon[:, 11]

    # compare mpl and bending stiffness for yaml, excel, subdyn and hawc2
    for i in range(2):
        # define mask depending on embedded monopile or monopile
        if i == 0:  # embedded monopile
            h2_vals = [mpl_h2_embmon, EIx_h2_embmon, EIy_h2_embmon]
            mask = twr_stn_yaml <= -30
            sd_vals = h2_vals # SubDyn only defined above mudline
        elif i == 1:  # monpile
            h2_vals = [mpl_h2_mon, EIx_h2_mon, EIy_h2_mon]
            mask = (twr_stn_yaml >= -30) & (twr_stn_yaml <= 15)
            sd_vals = [mpl_sd, EI_sd, EI_sd]
        
        # mask the yaml, excel, and subdyn values
        yaml_vals = [mpl_yaml[mask], EI_yaml[mask], EI_yaml[mask]]
        ex_vals = [mpl_ex[mask], EIx_ex[mask], EIy_ex[mask]]

        # compare the yaml, excel, subdyn, and hawc2 values
        for (arr_yaml, arr_ex, arr_sd, arr_h2) in zip(yaml_vals, ex_vals, sd_vals, h2_vals):
            np.testing.assert_allclose(arr_yaml, arr_ex, rtol=1e-3)  # yaml versus excel
            np.testing.assert_allclose(arr_yaml, arr_sd, rtol=1e-3)  # yaml versus subdyn
            np.testing.assert_allclose(arr_yaml, arr_h2, rtol=1e-3)  # yaml versus hawc2

def test_monopile_mass():
    """Check that the mass of the HAWC2 monopile matches what's expected."""

    # paths
    h2_embmon_path = FROOT / 'HAWC2/IEA-15-240-RWT-Monopile/data/IEA_15MW_RWT_Embeddedmonopile_st.dat'  # embedded monopile st file
    h2_mon_path = FROOT / 'HAWC2/IEA-15-240-RWT-Monopile/data/IEA_15MW_RWT_Monopile_st.dat'  # monopile st file

    # uncomment this section to recalculate yaml mass
    # yaml_path = FROOT / 'WT_Ontology/IEA-15-240-RWT.yaml'  # yaml file with data
    # yamldata = tstf.load_yaml(yaml_path)
    # twr_stn_yaml, out_diam_yaml, thick_yaml, E, G, rho, outfit = tstf.load_body_properties('monopile', yamldata)
    # mpl_yaml = tstf.calculate_mpl(out_diam_yaml, thick_yaml, rho, outfitting_factor=outfit)
    # mass_yaml = np.trapz(mpl_yaml, twr_stn_yaml) + 100e3
    mass_yaml = 1309947.640745313  # from -75 m to +15 m [kg]

    # load the hawc2 embedded-monopile tower properties
    h2_emdmon = tstf.load_hawc2_st(h2_embmon_path)
    twr_stn_embdmon = h2_emdmon[:, 0]
    mpl_h2_embmon = h2_emdmon[:, 1]

    # load the hawc2 monopile tower properties
    h2_mon = tstf.load_hawc2_st(h2_mon_path)
    twr_stn_mon = h2_mon[:, 0]
    mpl_h2_mon = h2_mon[:, 1]

    # calculate hawc2 mass
    mass_embmon = np.trapz(mpl_h2_embmon, twr_stn_embdmon)
    mass_mon = np.trapz(mpl_h2_mon, twr_stn_mon)
    mass_h2 = mass_embmon + mass_mon + 100e3

    assert abs((mass_h2 - mass_yaml) / mass_yaml) < 0.01
