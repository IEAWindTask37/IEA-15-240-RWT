    begin relative;
      body1  ifb 1 ;
      body2  tower 1;
      body2_eulerang 0.0 0.0 0.0;   same as global: zTT down, yTT downwind
    end relative;
;
    begin relative;
      body1  tower last;
      body2  towertop 1;
      body2_eulerang 0.0 0.0 0.0;   same as global: zTT down, yTT downwind
    end relative;
;
    begin relative;
      body1  towertop last;
      body2  connector 1;
      body2_eulerang 90.0 0.0 0.0; 
      body2_eulerang 6.0 0.0 0.0;    6 deg tilt; zC along shaft upwind, xC horizontal
    end relative;
;
    begin relative;
      body1  connector last;
      body2  shaft 1;
      body2_eulerang 0.0 0.0 0.0;    same as connector; zS along shaft upwind
;      body2_ini_rotvec_d1 0.0 0.0 -1.0 0.53 ; min 5 RPM / 0.524 rad/s, max 7.56 RPM, design TSR=9
    end relative;
;
    begin relative;
      body1  shaft last;         
      body2  hub1 1;
      body2_eulerang -90.0 0.0 0.0;    
      body2_eulerang 0.0 180.0 0.0;    
      body2_eulerang 4.0 0.0 0.0;      4 deg cone; zH along blade, xH towards LE
    end relative;
;
    begin relative;
      body1  shaft last;         
      body2  hub2 1;
      body2_eulerang -90.0 0.0 0.0;    
      body2_eulerang 0.0 60.0 0.0;   
      body2_eulerang 4.0 0.0 0.0;      4 deg cone angle
    end relative;
;
    begin relative;
      body1  shaft last;         
      body2  hub3 1;
      body2_eulerang -90.0 0.0 0.0;    
      body2_eulerang 0.0 -60.0 0.0;    
      body2_eulerang 4.0 0.0 0.0;      4 deg cone angle
    end relative;
;
    begin relative;
      body1  hub1 last;         
      body2  blade1 1;
      body2_eulerang 0.0 0.0 0;         same as hub; zB towards tip, xB towards LE
    end relative;
;
    begin relative;
      body1  hub2 last;         
      body2  blade2 1;
      body2_eulerang 0.0 0.0 0.0;    
    end relative;
;
    begin relative;
      body1  hub3 last;         
      body2  blade3 1;
      body2_eulerang 0.0 0.0 0.0;    
    end relative;
;
