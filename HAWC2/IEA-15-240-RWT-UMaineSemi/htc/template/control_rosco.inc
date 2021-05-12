   begin type2_dll;
     name  BL2H2 ;
     filename  ./control/rosco2hawc2/bl2h2.dll ;
     dll_subroutine_init name_initialize ;
     dll_subroutine_update name_update ;
     init_string ./control/rosco2hawc2/discon.dll ;
     arraysizes_init     6 1 ;
     arraysizes_update  14 1000 ;
     begin init;
       constant 1 0.01 ; 	time step [s]
       constant 2 1.0 ; 	Generator exchange ratio
       constant 3 0.0108 ;  P_loss/Pe_nom at   0%Pe -- scaled from NREL 5 MW
       constant 4 0.0204 ;  P_loss/Pe_nom at  50%Pe -- scaled from NREL 5 MW
       constant 5 0.0345 ; 	P_loss/Pe_nom at 100%Pe -- generator full load efficiency is 96.55% 
       constant 6 15000 ; 	Pe_nom [kW]
     end init;
;  
	begin output;
       general status ; 						 			1 Bladed like status
       general time ;   						 			2 Time                                                                                                     
       aero omega ; 							 			3 rotor sped in rad/s
	   constraint bearing2 pitch1 1 only 1; 	 			4 angle and angle velocity written to dll                                                     
       constraint bearing2 pitch2 1 only 1;		 			5 angle and angle velocity written to dll                                                     
       constraint bearing2 pitch3 1 only 1; 	 			6 angle and angle velocity written to dll                                                     
       wind free_wind_hor 1 0.0 0.0 -150 only 1; 			7 local wind at fixed position: coo (1=global,2=non-rotation rotor coo.), pos x, pos y, pos z 
       mbdy momentvec shaft 1 1 shaft only 3; 	 			8 shaft moment [kNm]
	   mbdy momentvec blade1 2  2 blade1 only 1; 			9 flap moment, blade 1 [kNm]
	   mbdy momentvec blade2 2  2 blade2 only 1; 			10 flap moment, blade 2 [kNm]
	   mbdy momentvec blade3 2  2 blade3 only 1; 			11 flap moment, blade 3 [kNm]
	   constraint bearing1 shaft_rot 1 only 1;				12 rotor azimuth angle [rad]
	   mbdy state acc tower 10 1.0 tower only 2;			13 tower top fore-aft acceleration [m/s^2]
	   mbdy state_rot omegadot tower 10 1.0 global only 1;	14 nacelle nodding accleration [rad/s^2]
    end output;
;
    begin actions;
      body moment_ext shaft 1 3;
    end actions;
   end type2_dll;
;
  begin type2_dll;  pitch servo
    name servo_with_limits ;
    filename  ./control/DTU_WEC/servo_with_limits.dll ;
    dll_subroutine_init init_servo_with_limits ;
    dll_subroutine_update update_servo_with_limits ;
    arraysizes_init  100 1 ;
    arraysizes_update  100 100 ;
	begin init ;
      constant 1   3    ; Number of blades [-]
      constant 2   1.0  ; Frequency of 2nd order servo model of pitch system [Hz]
      constant 3   0.7  ; Damping ratio 2nd order servo model of pitch system [-]
      constant 4   2.0  ; Max. pitch speed [deg/s]
      constant 5  15.0  ; Max. pitch acceleration [deg/s^2]
      constant 6   0.0  ; Min. pitch angle [deg] 
      constant  7 90.0  ; Max. pitch angle [deg]    
	  constant  8 -1    ; Time for pitch runaway [s]
	  constant  9 -1    ; Time for stuck blade 1 [s]
	  constant 10 0.0   ; Angle of stuck blade 1 [deg] (if > 90 deg then blade is stuck at instantaneous angle)
	end init ;
    begin output;
     general time ; 1
      dll type2_dll BL2H2 inpvec 2 ; 2
      dll type2_dll BL2H2 inpvec 3 ; 3
      dll type2_dll BL2H2 inpvec 4 ; 4
	  general constant 0           ; Flag for emergency pitch stop         [0=off/1=on]
    end output;           
;
    begin actions;    
      constraint bearing2 angle pitch1 ; Angle pitch1 bearing    [rad]
      constraint bearing2 angle pitch2 ; Angle pitch2 bearing    [rad]
      constraint bearing2 angle pitch3 ; Angle pitch3 bearing    [rad]
    end actions;                      
  end type2_dll;
;
begin type2_dll;  tower-blade-tip distance
  name towerclearance_mblade ;
  filename  ./control/DTU_WEC/towerclearance_mblade.dll ;
  dll_subroutine_init initialize ;
  dll_subroutine_update update ;
  arraysizes_init  3 1 ;
  arraysizes_update  15 6 ;
  begin init ;    Variables passed into initialization function
    constant  1 5.00  ; Tower radius at tower bottom [m] 
    constant  2 3.25  ; Tower radius at tower top [m]
    constant  3    3  ; Number of points to check [-]
  end init ;
  begin output;   Variables passed into update function
    mbdy state pos tower    1 0.0 global ; [1,2,3] global coordinates of tower base
    mbdy state pos tower   10 1.0 global ; [4,5,6] global coordinates of tower top
    mbdy state pos blade1  19 1.0 global ; [7,8,9] global coordinates of point 1 (blade 1 tip)
    mbdy state pos blade2  19 1.0 global ; [10,11,12] global coordinates of point 2 (blade 2 tip)
	mbdy state pos blade3  19 1.0 global ; [13,14,15] global coordinates of point 3 (blade 3 tip)
  end output;           
end type2_dll;
;
