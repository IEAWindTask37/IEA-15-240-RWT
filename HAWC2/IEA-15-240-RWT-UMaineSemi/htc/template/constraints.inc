     begin fix1;  towertop fixed to tower
		   body1 ifb 1 ;
		   body2 tower 1;
		 end fix1;
;
     begin fix1;  towertop fixed to tower
		   body1 tower last ;
		   body2 towertop 1;
		 end fix1;
;
     begin fix1;  connector fixed to towertop
		   body1 towertop last ;
		   body2 connector 1;
		 end fix1;
;
    begin bearing1;  shaft rotates as free bearing
     name  shaft_rot;
      body1 connector last;
      body2 shaft 1;
      bearing_vector 2 0.0 0.0 -1.0;  x=coo (0=global.1=body1.2=body2) vector in body2 coordinates where the free rotation is present
   end bearing1; 
;
     begin fix1;
		   body1 shaft last ;
		   body2 hub1 1;
		 end fix1;
;
     begin fix1;
		   body1 shaft last ;
		   body2 hub2 1;
		 end fix1;
;
     begin fix1;
		   body1 shaft last ;
		   body2 hub3 1;
		 end fix1; 
;	
    begin bearing2;
      name pitch1;		
      body1 hub1 last;
     body2 blade1 1;
			bearing_vector 2 0.0 0.0 -1.0;
   end bearing2;
;
    begin bearing2;
      name pitch2;		
      body1 hub2 last;
      body2 blade2 1;
			bearing_vector 2 0.0 0.0 -1.0;
    end bearing2;
;
    begin bearing2;
      name pitch3;		
      body1 hub3 last;
      body2 blade3 1;
			bearing_vector 2 0.0 0.0 -1.0;
    end bearing2;
