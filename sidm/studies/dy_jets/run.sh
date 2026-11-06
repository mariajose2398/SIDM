#!/bin/bash


python3 37_runDataBg.py --ch base_ljDisplacementIso  --hist lj_base --run_2mu --no-data --max_files_bg -1
python3 37_runDataBg.py --ch base_ljObjCut_ljIso_1lj  --hist displacement_base --run_2mu --no-data --max_files_bg -1
python3 37_runDataBg.py --ch baseNoLj  --hist isolation_base --run_2mu --no-data --max_files_bg -1



x