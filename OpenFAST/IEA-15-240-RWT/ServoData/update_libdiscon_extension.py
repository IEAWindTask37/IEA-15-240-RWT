import glob
import os
from rosco import discon_lib_path

if __name__ == "__main__":
    this_dir   = os.path.dirname(os.path.abspath(__file__))
    servo_list = glob.glob(os.path.join(this_dir, '../../*/*Servo*.dat'))

    for ifile in servo_list:
        # Read in current ServoDyn file
        with open(ifile, "r") as f:
            lines = f.readlines()

        # Write correction
        with open(ifile, "w") as f:
            for line in lines:
                if line.find("DLL_FileName") >= 0:
                    f.write(f"\"{discon_lib_path}\"    DLL_FileName - Name/location of the dynamic library (.dll [Windows] or .so [Linux]) in the Bladed-DLL format (-) [used only with Bladed Interface]\n")
                else:
                    f.write(line)
                
