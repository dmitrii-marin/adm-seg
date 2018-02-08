#!/bin/bash
# A sample Bash script, by Ryan
echo Hello World!
./run_pascal_scribble.sh 9000
matlab -nodisplay -nosplash -r "cd ~/Disney/interactive/;run normalizedcutrefinebatch;exit"
./run_pascal_scribble_iterative.sh
