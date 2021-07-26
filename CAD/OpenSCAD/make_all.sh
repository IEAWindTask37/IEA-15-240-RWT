#!/bin/bash

froot="IEA-15-240-RWT"
float="${froot}_VolturnUS-S"
fixed_fine="../../WT_Ontology/${froot}_FineGrid.yaml"
float_fine="../../WT_Ontology/${float}_FineGrid.yaml"
openscad="/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"

# Blade
python -m windio2cad --input $fixed_fine --output ${froot}_blade.stl --openscad $openscad --blade
/bin/mv intermediate.scad ${froot}_blade.scad

# Tower
python -m windio2cad --input $fixed_fine --output ${froot}_tower.stl --openscad $openscad --tower
/bin/mv intermediate.scad ${froot}_tower.scad

# Monopile
python -m windio2cad --input $fixed_fine --output ${froot}_monopile.stl --openscad $openscad --monopile
/bin/mv intermediate.scad ${froot}_monopile.scad

# Floater
python -m windio2cad --input $float_fine --output ${float}_floater.stl --openscad $openscad --floater
/bin/mv intermediate.scad ${float}_floater.scad

# Full fixed-bottom
python -m windio2cad --input $fixed_fine --output ${froot}.stl --openscad $openscad
/bin/mv intermediate.scad ${froot}.scad

# Full floating
python -m windio2cad --input $float_fine --output ${float}.stl --openscad $openscad
/bin/mv intermediate.scad ${float}.scad

# Compress all of the large files
/bin/ls *.stl *.scad | while read file; do zip ${file}.zip $file; done;
