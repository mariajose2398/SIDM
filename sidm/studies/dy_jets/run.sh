#!/bin/bash


python3 37_runDataBg.py --ch cr_Zmumu_noLJ_noVeto --hist dy_jets_new --run_2mu --bgs DYJetsToLL_M10to50 DYJetsToLL_M50
python3 37_runDataBg.py --ch base_ljIso --hist dy_jets_new --no-data --no-bgs --run_2mu
python3 37_runDataBg.py --ch base_ljObjCut_ljIso_2lj --hist displacement_base ----max_files_bg -1 --no-data --run_2mu


