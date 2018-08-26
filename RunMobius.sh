#!/bin/bash

source /etc/profile.d/modules.sh
module load cse/conda27
cd $PWD
echo "Current directory  = " $PWD
echo "Running Mobius Job = " $2
#/share/apps/conda27/anaconda2/bin/python < MobiusHoughFiles.py ${1} ${2}
#/share/apps/conda27/anaconda2/bin/python MobiusHoughFiles.py $1 $2 $3
python MobiusHoughFiles.py $1 $2 $3
