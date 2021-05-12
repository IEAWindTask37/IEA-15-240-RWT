;
  begin type2_dll;  dtu basic controller
    name dtu_we_controller ;
;    filename  ./control/dtu_wec/dtu_we_controller.dll ;
	filename  ./control/dtu_wec/dtu_we_controller_NF.dll ; DLL with nacelle feedback implemented in the pitch controller loop
    dll_subroutine_init init_regulation_advanced ;
    dll_subroutine_update update_regulation ;
    arraysizes_init  100 1 ;
    arraysizes_update  100 100 ;
;   used in down regulation control strategies and wind speed estimation
;		init_string ./control/additional_control_parameters.txt;
	begin init ;
; 	  Overall parameters
      constant   1 15000; 			Rated power [kW]                         
      constant   2 0.524; 			Minimum rotor (LSS) speed [rad/s]
      constant   3 0.792; 			Rated rotor (LSS) speed [rad/s]
      constant   4 21586451.33303; 	Maximum allowable generator torque [Nm]
      constant   5 100.0; 			Minimum pitch angle, theta_min [deg], 
						; 			if |theta_min|>90, then a table of <wsp,theta_min> is read ;
						; 			from a file named 'wptable.n', where n=int(theta_min)
      constant   6 90.0; 			Maximum pitch angle [deg]
      constant   7 2.0; 			Maximum pitch velocity operation [deg/s]
      constant   8 1.0081; 			Frequency of generator speed filter [Hz]
      constant   9 0.7; 			Damping ratio of speed filter [-]
      constant  10 1.01;		 	Frequency of free-free DT torsion mode [Hz], if zero no notch filter used
;	  Partial load control parameters
      constant  11 30228800; 		Optimal Cp tracking K factor [Nm/(rad/s)^2], ;
                           ;	 	Qg=K*Omega^2, K=eta*0.5*rho*A*Cp_opt*R^3/lambda_opt^3                     
      constant  12 0.112427E+09; 		Proportional gain of torque controller [Nm/(rad/s)]
      constant  13 0.201829E+08; 		Integral gain of torque controller [Nm/rad]
      constant  14 0.0; 			Differential gain of torque controller [Nm/(rad/s^2)]
;     Full load control parameters
      constant  15   0;				Generator control switch [1=constant power, 0=constant torque]
      constant  16   0.96119; 		Proportional gain of pitch controller [rad/(rad/s)]
      constant  17   0.640241E+00; 	Integral gain of pitch controller [rad/rad]
      constant  18   0.0;		 	Differential gain of pitch controller [rad/(rad/s^2)]
      constant  19   0.4e-8; 		Proportional power error gain [rad/W]
      constant  20   0.4e-8; 		Integral power error gain [rad/(Ws)]
      constant  21 	 11.95434;		Coefficient of linear term in aerodynamic gain scheduling, KK1 [deg]
      constant  22 	 720.25183; 	Coefficient of quadratic term in aerodynamic gain scheduling, KK2 [deg^2] &
							 ; 		(if zero, KK1 = pitch angle at double gain)
      constant  23   1.5; 			Relative speed for double nonlinear gain [-]
;     Cut-in simulation parameters
      constant  24  -1; 			Cut-in time [s], no cut-in is simulated if zero or negative
      constant  25  1.0; 			Time delay for soft start of torque [1/1P]
;     Cut-out simulation parameters
      constant  26  -1; 			Shut-down time [s], no shut-down is simulated if zero or negative
      constant  27  5.0; 			Time of linear torque cut-out during a generator assisted stop [s]
      constant  28  1; 				Stop type [1=normal, 2=emergency]
      constant  29  1.0; 			Time delay for pitch stop after shut-down signal [s]
      constant  30  2.0; 			Maximum pitch velocity during initial period of stop [deg/s]
      constant  31  3.0; 			Time period of initial pitch stop phase [s] (maintains pitch speed specified in constant 30)
      constant  32  2.0; 			Maximum pitch velocity during final phase of stop [deg/s]
;     Expert parameters (keep default values unless otherwise given)
      constant  33   2.0;			Time for the maximum torque rate = Maximum allowable generator torque/(constant 33 + 0.01s) [s]
      constant  34   2.0;			Upper angle above lowest minimum pitch angle for switch [deg], if equal then hard switch
      constant  35  95.0;			Percentage of the rated speed when the torque limits are fully opened [%]
      constant  36   2.0;			Time constant of 1st order filter on wind speed used for minimum pitch [1/1P]
      constant  37   1.0;			Time constant of 1st order filter on pitch angle used for gain scheduling [1/1P]
;     Drivetrain damper
      constant  38   0.0; 			Proportional gain of active DT damper [Nm/(rad/s)], requires frequency in input 10
;	  Over speed
	  constant  39  50.0; 			Overspeed percentage before initiating turbine controller alarm (shut-down) [%]
;     Additional non-linear pitch control term (not used when all zero)
	  constant  40   0.0; 			Rotor speed error scaling factor [rad/s]
	  constant  41   0.0; 			Rotor acceleration error scaling factor [rad/s^2]
	  constant  42   0.0; 			Pitch rate gain [rad/s]
;     Storm control command
	  constant  43   28.0; 			Wind speed 'Vstorm' above which derating of rotor speed is used [m/s]
	  constant  44   28.0; 			Cut-out wind speed (only used for derating of rotor speed in storm) [m/s]	  
;     Safety system parameters
	  constant  45   50.0;			Overspeed percentage before initiating safety system alarm (shut-down) [%]
	  constant  46    2.0;			Max low-pass filtered tower top acceleration level [m/s^2]
;     Turbine parameter
	  constant  47  240.0; 			Nominal rotor diameter [m]
;     Parameters for rotor inertia reduction in variable speed region
      constant  48    0.0; 			Proportional gain on rotor acceleration in variable speed region [Nm/(rad/s^2)] (not used when zero)
;     Parameters for alternative partial load controller with PI regulated TSR tracking
      constant  49    9.0; 			Optimal tip speed ratio [-] (only used when K=constant 11 = 0 otherwise  Qg=K*Omega^2 is used)
;     Parameters for adding aerodynamic drivetrain damping on gain scheduling
      constant  50    0.0;			Aerodynamic DT damping coefficient at the operational point of zero pitch angle [Nm/(rad/s)] (not used when zero)
      constant  51    0.0;			Coefficient of linear term in aerodynamic DT damping scheduling, KK1 [deg]
      constant  52    0.0;			Coefficient of quadratic term in aerodynamic DT damping scheduling, KK2 [deg^2]
;     Torque exclusion zone
      constant  53    0.0; 			Exclusion zone: Lower speed limit [rad/s] (Default 0 used if zero)	  
      constant  54    0.0; 			Exclusion zone: Generator torque at lower limit [Nm] (Default 0 used if zero)	  
      constant  55    0.0; 			Exclusion zone: Upper speed limit [rad/s] (if =< 0 then exclusion zone functionality is inactive)               	  
      constant  56    0.0; 			Exclusion zone: Generator torque at upper limit [Nm] (Default 0 used if zero) 	  
      constant  57    0.0; 			Time constant of reference switching at exclusion zone [s] (Default 0 used if zero)	  
;     DT torsion mode damper	  
      constant  58    0.0; 			Frequency of notch filter [Hz] (Default 10 x input 10 used if zero)	  
      constant  59    0.0; 			Damping of BP filter [-] (Default 0.02 used if zero) 	  
      constant  60    0.0; 			Damping of notch filter [-] (Default 0.01 used if zero) 	  
      constant  61    0.0; 			Phase lag of damper [s] =>  max 40*dt (Default 0 used if zero) 	  
;     Fore-aft Tower mode damper	  
      constant  62    0.0;		    Frequency of BP filter [Hz] (Default 10 used if zero)\\ platform pitching natural frequency 	  
      constant  63    0.0;		Frequency of notch fiter [Hz] (Default 10 used if zero)\\ 	  
      constant  64    0.0;			Damping of BP filter [-] (Default 0.02 used if zero)\\	  
      constant  65    0.0;			Damping of notch filter [-] (Default 0.01 used if zero)\\	  
      constant  66    0.0;			Gain of damper [-] (Default 0 used if zero)\\ 	  
      constant  67    0.0;			Phase lag of damper [s] =>  max 40*dt (Default 0 used if zero)\\ 	  
      constant  68    0.0;			Time constant of 1st order filter on PWR used for fore-aft Tower mode damper GS [Hz] (Default 10 used if zero)	  
      constant  69    0.0;			Lower PWR limit used for fore-aft Tower mode damper GS [-] (Default 0 used if zero)	  
      constant  70    0.0;			Upper PWR limit used for fore-aft Tower mode damper GS [-] (Default 0 used if zero) 	  
;     Side-to-side Tower mode filter	  
      constant  71    0.0; 			Frequency of Tower side-to-sede notch filter [Hz] (Default 100 used if zero)	  
      constant  72    0.0; 			Damping of notch filter [-] (Default 0.01 used if zero)	  
      constant  73    0.0; 			Max low-pass filtered tower top acceleration level before initiating safety system alarm (shut-down) [m/s^2] (Default 1.1 x input 46 used if zero)	  
      constant  74    0.0; 			Time constant of 1st order filter on tower top acceleration [1/1P] (Default 1 used if zero)	  
;     Pitch deviation monitor parameters	  
      constant  75 1005020; 		Parameters for pitch deviation monitoring. The format is 1,nnn,mmm 	  
                          ; 		where 'nnn' [s] is the period of the moving average and 'mmm' is threshold of the deviation [0.1 deg] (functionality is inactive if value $<$ 1,000,000)	  
;     Gear ratio	  
      constant  76    1.0; 			Gear ratio used for the calculation of the LSS rotational speeds and the HSS generator torque reference [-] (Default 1 if zero)	  
;
; HERE STARTS CONSTANTS FOR NACELLE FEEDBACK FEATURE OF DTU WEC - Uncomment contants 77-94 if the nacelle feedback feature is used
;
;     Rotor speed notch filter for constant power operation above rated
	  constant  77 	  0.0; 			Frequency of notch filter [Hz] applied on the rotor speed before computing torque above rated (constant power), if zero no notch filter used
	  constant 	78 	  0.0; 			Damping of notch filter [-] applied on the rotor speed before computing torque above rated (constant power), (Default 0.01 used if zero)
;     De-rating strategies
	  constant 	79 		0; 			Derate strategy. 0 = No Derating, 1 = constant rotation, 2 = max rotation  3 = min ct
	  constant  80 	  100; 			Derate percentage (eg. 70 means 70% of nominal power)
	  constant 	81    0.0; 			Moment of inertia of the rotor 
; 	  Floating additional control loop
	  constant 	82    0.0;    		Rated wind speed (Only used when constant 79 = 3 OR constant 91/92/93/94 != 0)
	  constant 	83    0.0619;		    Gain for the loop mapping from tower velocity to blade pitch [rad/(m/s)]
	  constant  84    0.0; 			Gain for the loop mapping from tower velocity to GenTorque [Nm/(m/s)]
	  constant  85   10.0; 			Time to switch on the floating control loop [s]
	  constant  86    0.036; 		Frequency of LP filter [Hz] (Default 0 if filter not used) - chose properly
	  constant  87    1.0; 			Damping ratio of LP filter  [-]
	  constant  88    0.497; 		Frequency of BP filter [Hz] (Default 0 if filter not used)
	  constant  89    1.0; 			Damping ratio of BP filter  [-] (Default 0.02)
	  constant  90    0.0; 			Time constant of BP filter [s] (Default 0)
	  constant  91    0.0; 			Coefficient of linear term in gain scheduling for tower-pitch loop, KK1 [1/(m/s)];
									; Kgain * (1 + KK1 *abs(WSPfilt - WSPrated) + KK2 * abs(WSPfilt - WSPrated)**2)
	  constant  92    0.0; 			Coefficient of quadratic term in gain scheduling for tower-pitch loop, KK2 [1/(m/s)^2];
	  constant  93    0.0; 			Coefficient of linear term in gain scheduling for tower-genTorq loop, KK1 [1/(m/s)];
	  constant  94    0.0; 			Coefficient of quadratic term in gain scheduling for tower-genTorq loop, KK2 [1/(m/s)^2];
	end init ;
;
    begin output ;
      general time ; [s]     
      constraint bearing1 shaft_rot 1 only 2 ; 		Drivetrain speed [rad/s]
      constraint bearing2 pitch1 1 only 1 ; 		[rad]         
      constraint bearing2 pitch2 1 only 1 ; 		[rad]                               
      constraint bearing2 pitch3 1 only 1 ; 		[rad]                               
      wind free_wind 1 0.0 0.0 -150 ; 		    	Global coordinates at hub height
      dll type2_dll generator_servo inpvec 2 ; 		Elec. power from generator servo .dll
      dll type2_dll generator_servo inpvec 8 ; 		Grid state flag from generator servo .dll
;      mbdy state acc towertop 1 1 global only 1 ;   Tower top x-acceleration [m/s^2]
;      mbdy state acc towertop 1 1 global only 2 ;   Tower top y-acceleration [m/s^2]
	  ; - Only with nacelle feedback - ;
      mbdy state acc towertop 1 1 global only 2 ;   Tower top x-acceleration [m/s^2]
      mbdy state acc towertop 1 1 global only 1 ; 	Tower top y-acceleration [m/s^2]
	  general constant 0 ;						 	IPC - if 0, no IPC action is done
	  general constant 0 ;						 	IPC - if 0, no IPC action is done
	  general constant 0 ;						 	IPC - if 0, no IPC action is done
	  mbdy state vel towertop 1 1 global only 2;	tower-top fore-aft velocity [m/s]
    end output;    
  end type2_dll;
;
   begin type2_dll;  generator servo
     name generator_servo ;
     filename  ./control/dtu_wec/generator_servo.dll ;
     dll_subroutine_init init_generator_servo ;
     dll_subroutine_update update_generator_servo ;
     arraysizes_init  100 1 ;
     arraysizes_update 100 100 ;
 	begin init ;
       constant 1  20.0    ; Frequency of 2nd order servo model of generator-converter system [Hz]   
       constant 2  0.9     ; Damping ratio 2nd order servo model of generator-converter system [-]
       constant 3 21586451.33303 ; Maximum allowable LSS torque (pull-out torque) [Nm]
       constant 4 0.9655 ; Generator efficiency [-]
       constant 5 1.0      ; Gearratio [-]
       constant 6 0.0      ; Time for half value in softstart of torque [s] 
       constant 7 -1       ; Time for grid loss [s] (never if lower than zero)
     end init ;
;
     begin output;
       general time                             ;   Time [s]    
       dll type2_dll dtu_we_controller inpvec 1 ;   Electrical torque reference [Nm]  
       constraint bearing1 shaft_rot 1 only 2   ;   Generator LSS speed [rad/s]   
       mbdy momentvec shaft 1 1 shaft only 3    ;   Shaft moment [kNm] (Qshaft)
     end output;
;
     begin actions;    
        mbdy moment_int shaft 1 -3 shaft connector 2 ;   Generator LSS torque [Nm]
     end actions;
   end type2_dll;
;
   begin type2_dll;  mechanical brake
     name mech_brake ;
     filename  ./control/dtu_wec/mech_brake.dll ;
     dll_subroutine_init init_mech_brake ;
     dll_subroutine_update update_mech_brake ;
     arraysizes_init    100 1 ;
     arraysizes_update  100 100 ;
 	begin init ;
      constant 1 12951870.799818 ; Fully deployed maximum brake torque [Nm] (0.6*max torque)
      constant 2     100.0   ; Parameter alpha used in Q = tanh(omega*alpha), typically 1e2/Omega_nom
      constant 3       0.5   ; Delay time for before brake starts to deploy [s]
      constant 4       0.6   ; Time for brake to become fully deployed [s]
     end init ;
;
     begin output;
	   general time                                 ; Time [s]
	   constraint bearing1 shaft_rot 1 only 2       ; Generator LSS speed [rad/s]
	   dll type2_dll dtu_we_controller inpvec 25    ; Command to deploy mechanical disc brake [0,1]
     end output;
;
     begin actions;    
        mbdy moment_int shaft 1 -3 shaft connector 2 ;   Brake LSS torque [Nm]
     end actions;
   end type2_dll;
;
  begin type2_dll;  pitch servo
    name servo_with_limits ;
    filename  ./control/dtu_wec/servo_with_limits.dll ;
    dll_subroutine_init init_servo_with_limits ;
    dll_subroutine_update update_servo_with_limits ;
    arraysizes_init  100 1 ;
    arraysizes_update  100 100 ;
	begin init ;
      constant 1   3    ; Number of blades [-]
      constant 2   1.0  ; Frequency of 2nd order servo model of pitch system [Hz]
      constant 3   0.7  ; Damping ratio 2nd order servo model of pitch system [-]
      constant 4   3    ; Max. pitch speed [deg/s]
      constant 5  15.0  ; Max. pitch acceleration [deg/s^2]
      constant 6   0.0  ; Min. pitch angle [deg] 
      constant  7 90.0  ; Max. pitch angle [deg]    
	  constant  8 -1    ; Time for pitch runaway [s]
	  constant  9 -1    ; Time for stuck blade 1 [s]
	  constant 10 0.0   ; Angle of stuck blade 1 [deg] (if > 90 deg then blade is stuck at instantaneous angle)
	end init ;
    begin output;
      general time        ;  Time                         [s]     
       dll type2_dll dtu_we_controller inpvec 2     ;  Pitch1 demand angle          [rad]
       dll type2_dll dtu_we_controller inpvec 3     ;  Pitch2 demand angle          [rad]
       dll type2_dll dtu_we_controller inpvec 4     ;  Pitch3 demand angle          [rad]
       dll type2_dll dtu_we_controller inpvec 26    ;  Flag for emergency pitch stop         [0=off/1=on]
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
  filename  ./control/dtu_wec/towerclearance_mblade.dll ;
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
    mbdy state pos blade1  19 1.0 global  ; [7,8,9] global coordinates of point 1 (blade 1 tip)
    mbdy state pos blade2  19 1.0 global  ; [10,11,12] global coordinates of point 2 (blade 2 tip)
	mbdy state pos blade3  19 1.0 global  ; [13,14,15] global coordinates of point 3 (blade 3 tip)
  end output;           
end type2_dll;
;
