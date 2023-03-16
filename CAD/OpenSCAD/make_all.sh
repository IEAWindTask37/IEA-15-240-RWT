#!/bin/bash

froot="IEA-15-240-RWT"
float="${froot}_VolturnUS-S"
fixed_yaml="../../WT_Ontology/${froot}.yaml"
float_yaml="../../WT_Ontology/${float}.yaml"
openscad="/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"

# Blade
python -m windio2cad --input $fixed_yaml --output ${froot}_blade.stl --openscad $openscad --blade
/bin/mv intermediate.scad ${froot}_blade.scad

# Tower
python -m windio2cad --input $fixed_yaml --output ${froot}_tower.stl --openscad $openscad --tower
/bin/mv intermediate.scad ${froot}_tower.scad

# Floating Tower
python -m windio2cad --input $float_yaml --output ${float}_tower.stl --openscad $openscad --tower
/bin/mv intermediate.scad ${float}_tower.scad

# Monopile
python -m windio2cad --input $fixed_yaml --output ${froot}_monopile.stl --openscad $openscad --monopile
/bin/mv intermediate.scad ${froot}_monopile.scad

# Floater
python -m windio2cad --input $float_yaml --output ${float}_floater.stl --openscad $openscad --floater
/bin/mv intermediate.scad ${float}_floater.scad

# Full fixed-bottom
python -m windio2cad --input $fixed_yaml --output ${froot}.stl --openscad $openscad
/bin/mv intermediate.scad ${froot}.scad

# Full floating
python -m windio2cad --input $float_yaml --output ${float}.stl --openscad $openscad
/bin/mv intermediate.scad ${float}.scad

# Compress all of the large files
/bin/ls *.stl *.scad | while read file; do zip ${file}.zip $file; done;
