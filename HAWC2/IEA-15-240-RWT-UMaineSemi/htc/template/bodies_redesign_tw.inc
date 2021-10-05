  begin main_body;  tower
    name        tower ; 
    type        timoschenko ;
    nbodies     1 ;
    node_distribution     c2_def ;
    damping_posdef   0.0 0.0 0.0 0.000982 0.000982 0.0001011  ; tuned to 2% log dec on 1st FA/SS/torsion modes (#1, #2, #7)
     begin timoschenko_input;
      filename ./data/IEA_15MW_RWT_UMaine_Tower_st_redesign.dat;
      set 1 1 ; 
    end timoschenko_input;
    begin c2_def;              Definition of centerline (main_body coordinates)
      nsec 11;
      sec	 1	0	0	 0.0000e+00    0	;  x,y,z,twist
      sec	 2	0	0	-1.3000e+01    0	;
      sec	 3	0	0	-2.6000e+01    0	;
      sec	 4	0	0	-3.9000e+01    0	;
      sec	 5	0	0	-5.2000e+01    0	;
      sec	 6	0	0	-6.5000e+01    0	;
      sec	 7	0	0	-7.8000e+01    0	;
      sec	 8	0	0	-9.1000e+01    0	;
      sec	 9	0	0	-1.0400e+02    0	;
      sec	10	0	0	-1.1700e+02    0	;
      sec	11	0	0	-1.2958e+02    0	;
     end c2_def ;
    end main_body;
;
  begin main_body;  towertop
    name        towertop ;
    type        timoschenko ;
    nbodies     1 ;
    node_distribution     c2_def ;
    damping_posdef  0.0  0.0  0.0  7.00E-04  7.00E-04  7.00E-04  ;   dummy values (stiff body)
	concentrated_mass  1  0.00  0.00 0.00  1.000e+05  0.00  0.00  0.00	;  yaw system
	concentrated_mass  1  0.00  -4.688  -4.240  5.309e+05  7.674778e+06  1.055686e+07  8.127143e+06	;  nacelle: NR+R mass; NR inertia
	begin timoschenko_input;
      filename ./data/IEA_15MW_RWT_Towertop_st.dat ;
      set 1 1 ;                
    end timoschenko_input;
    begin c2_def;
      nsec 2;
      sec 1 0.0 0.0  0.0    0.0 ; x,y,z,twist
      sec 2 0.0 0.0 -4.3478  0.0 ; 
    end c2_def ;
  end main_body;
;
  begin main_body;  connector
    name        connector ;
    type        timoschenko ;
    nbodies     1 ;
    node_distribution     c2_def ;
    damping_posdef  0.0  0.0  0.0  7.00E-04  7.00E-04  7.00E-04  ;   dummy values (stiff body)
	begin timoschenko_input;
      filename ./data/IEA_15MW_RWT_Connector_st.dat ;
      set 1 1 ;                
    end timoschenko_input;
    begin c2_def;
      nsec 2;
      sec 1 0.0 0.0  0.0    0.0 ; x,y,z,twist
      sec 2 0.0 0.0  5.96769163920947  0.0 ; 
    end c2_def ;
  end main_body;
;
  begin main_body;  shaft
    name        shaft ;
    type        timoschenko ;
    nbodies     1 ;
    node_distribution     c2_def ;
	damping_posdef  0.0 0.0 0.0 4.65E-04  4.65E-04  3.078839e-04 ;  Kx=Ky=dummy; Kz tuned to 5% critical for free-free Ig, Ir
    concentrated_mass  1  0.0 0.0 0.0 0.0 0.0 0.0 3.222631e+06	;	generator inertia about shaft
    concentrated_mass  2  0.0 0.0 0.0 1.90000e+05 0.0 0.0 1.373471e+06	;	hub mass/inertia;
	begin timoschenko_input;
      filename ./data/IEA_15MW_RWT_Shaft_st.dat ;
      set 1 1 ;
    end timoschenko_input;
    begin c2_def;
      nsec 2;
      sec 1 0.0 0.0  0.00 0.0 ; x,y,z,twist
      sec 2 0.0 0.0  5.168312556956474 0.0 ;
    end c2_def ;
  end main_body;	
;
  begin main_body;  hub
    name        hub1 ;              
    type        timoschenko ;
    nbodies     1 ;
    node_distribution     c2_def ;
    damping_posdef  0.0  0.0  0.0  3.00E-06  3.00E-06  2.00E-05;  dummy values (rigid)
	begin timoschenko_input;
      filename ./data/IEA_15MW_RWT_Hub_st.dat ;
      set 1 1 ;                
    end timoschenko_input;
    begin c2_def;
      nsec 2;
      sec 1 0.0 0.0 0.0 0.0 ; x,y,z,twist
      sec 2 0.0 0.0 3.0 0.0 ; 
    end c2_def ;
  end main_body;
;
  begin main_body;
    name           hub2 ;
    copy_main_body hub1;
  end main_body;
;
  begin main_body;
    name           hub3 ;
    copy_main_body hub1 ;
  end main_body;
;
  begin main_body; blade
    name        blade1 ;        
    type        timoschenko ;
    nbodies     10 ;
    node_distribution    c2_def;
    damping_aniso  0.0 0.0 0.0 2.955e-3 2.424e-3 1.0e-8 ; blade damping tuned to 3% log dec flap/edge, torsion arbitrarily small
    begin timoschenko_input ;
      filename ./data/IEA_15MW_RWT_Blade_st.dat;
      FPM 1;  using fully populated matrix from BECAS
      set 1 1 ;
    end timoschenko_input;
    begin c2_def;
      nsec 20;
      sec  1                       0.0                      0.0                       0.0 -1.559455301971172e+01;
      sec  2 -1.947632306428538e-01  1.118894130288145e-01  6.201182297139584e+00 -1.517563341692140e+01;
      sec  3 -4.980721139630508e-01  2.670165808116275e-01  1.307124721605084e+01 -1.295380896003360e+01;
      sec  4 -8.093831584098645e-01  3.739824412925188e-01  2.055236308990396e+01 -9.768502125620273e+00;
      sec  5 -9.937536784943122e-01  3.756138793475129e-01  2.854529515461813e+01 -7.218598749599913e+00;
      sec  6 -9.933625716774784e-01  3.279577406655898e-01  3.690610036286960e+01 -5.088885094904810e+00;
      sec  7 -9.420208165255064e-01  2.681933097079662e-01  4.546140184592034e+01 -3.424371460144447e+00;
      sec  8 -9.035358795700276e-01  9.920060012536037e-02  5.402214442561254e+01 -2.220313133432046e+00;
      sec  9 -8.447578575993667e-01 -1.506239443252051e-01  6.239937510471066e+01 -1.268966518616668e+00;
      sec 10 -7.733898129195858e-01 -4.966218792643505e-01  7.041842166402073e+01 -5.370306489608649e-01;
      sec 11 -6.939710236584598e-01 -9.203225799218611e-01  7.793519479032008e+01  9.353172039816750e-02;
      sec 12 -6.146949665482084e-01 -1.345973550362133e+00  8.484691997117963e+01  8.493766337535498e-01;
      sec 13 -5.437000694616229e-01 -1.784306455815415e+00  9.108729127572076e+01  1.672206858318514e+00;
      sec 14 -4.808082504753410e-01 -2.204478814353722e+00  9.663141565650780e+01  2.143433636129829e+00;
      sec 15 -4.212886122526527e-01 -2.589840487843239e+00  1.014866648892364e+02  2.171780282338360e+00;
      sec 16 -3.683459920519034e-01 -2.945714468488638e+00  1.056843042126973e+02  2.088794131302387e+00;
      sec 17 -3.232741746814742e-01 -3.266603285702173e+00  1.092738898405172e+02  1.934836373952981e+00;
      sec 18 -2.879473566064166e-01 -3.550229846431562e+00  1.123151327495461e+02  1.723367849618178e+00;
      sec 19 -2.033818943504264e-01 -3.794043032744452e+00  1.148691506459029e+02  1.487742695689245e+00;
      sec 20 -6.589359677411839e-02 -4.001429044657352e+00  1.170000000000000e+02  1.242387706272969e+00;
     end c2_def ;
   end main_body;
;
  begin main_body;
    name           blade2 ;
    copy_main_body blade1;
  end main_body;
;
  begin main_body;
    name           blade3 ;
    copy_main_body blade1 ;
  end main_body;
