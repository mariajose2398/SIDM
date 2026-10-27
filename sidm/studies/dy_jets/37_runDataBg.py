print("IMPORTING MODULES")

import os, sys, subprocess, tempfile
from coffea import processor
import coffea.util
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
sidm_path = str(os.getcwd()).split('/sidm')[0]
if sidm_path not in sys.path:
    sys.path.insert(1, sidm_path)

from sidm.tools import utilities, scaleout, sidm_processor, llpnanoaodschema
import argparse

bgs = [
     "DYJetsToLL_M10to50",
      "DYJetsToLL_M50",
       "TTJets",
      "QCD_Pt1000",
      "QCD_Pt800To1000",
      "QCD_Pt600To800",
      "QCD_Pt470To600",
      "QCD_Pt300To470",
      "QCD_Pt170To300",
      "QCD_Pt120To170",
       "QCD_Pt80To120",
       "QCD_Pt50To80",
       "QCD_Pt30To50",
      "QCD_Pt20To30", #error
      "QCD_Pt15To20", #error
      "WZ", 
      "ZZ",
      "WW"
        ]

data = [ 
'DoubleMuon_2018C',
]


parser = argparse.ArgumentParser()
parser.add_argument( "--ch" , type=str, default="base_ljIso",help="Channel name")
parser.add_argument("--hist", type=str, default="deltaR_cosmic",help="Histogram collection name")
parser.add_argument("--no-data", action="store_true",help="Skip data samples")
parser.add_argument("--no-bgs", action="store_true", help="Skip background samples")
parser.add_argument("--max_files_bg", type=int, default=1000)
parser.add_argument("--max_files_data", type=int, default=-1)
parser.add_argument("--max_files_signal", type=int, default=-1)
parser.add_argument("--run_2mu", action="store_true", help="Run 2mu signal samples")
parser.add_argument("--run_4mu", action="store_true", help="Run 4mu signal samples")
parser.add_argument("--masses", type=int, nargs="+", default=[500], help="Bound-state masses (GeV)")
parser.add_argument("--bgs", nargs="+", choices=bgs, default=bgs)
args = parser.parse_args()


bgs = args.bgs
max_files_bg = args.max_files_bg
max_files_data = args.max_files_data
max_files_signal = args.max_files_signal

args = parser.parse_args()
print("VOMSPROXY")
os.environ["X509_USER_PROXY"] = "/uscms/home/mjose/x509up_u59701"
scaleout.check_voms_proxy()

print("CREATE CLUSTER CLIENR")
cluster, client = scaleout.make_lpc_client(
    min_workers=10,
    max_workers=100,
    memory='4GB',
    disk='4GB',
    scheduler_options={"dashboard_address": ":8790"}
)
print('dashboard:', cluster.dashboard_link)
client.wait_for_workers(1, timeout=600)
print('first worker connected; cluster:', cluster)


print("CREATE RUNNER")
runner = processor.Runner(
    executor=processor.DaskExecutor(client=client, status=False),
    schema=llpnanoaodschema.LLPNanoAODSchema,
    skipbadfiles=True,
    chunksize=10000,
)

print("CHANNEL NAME AND HIST COLLECTION")
channel_name = args.ch
hist_collection = args.hist

print(f"channel_name:{channel_name}")

print("Processor creation")
p = sidm_processor.SidmProcessor(
    [channel_name],
    [ hist_collection ], 
    # unweighted_hist=True,
)

max_files_bg = 1000
max_files_data = -1
max_files_signal = -1



if args.no_data:
    data = []
print("Processing DATA")
Fail_list = []
file_names = []
for x in data:
    print(x)
    try:
         fileset = utilities.make_fileset([x],'llpNanoAOD_v2',location_cfg='data_skimmed.yaml',
                                      max_files=max_files_data,replace_xcache=True,)
         output_data = runner.run(fileset, treename='Events', processor_instance=p)
    except Exception as e:
         print(f"failed_running {x}, skipping")
         print(e)
         Fail_list.append(x)
         continue
    date_folder = datetime.now().strftime("%Y_%d_%m")
    dir_path = f"OutputFiles/{channel_name}/{date_folder}"
    os.makedirs(dir_path, exist_ok=True)
    filename =  f"{dir_path}/{x}.coffea"
    coffea.util.save(output_data, filename )
    file_names.append(filename)


print("Processing background")
if args.no_bgs:
    bgs = []
for x in bgs:
    print(x)
    try:
         fileset = utilities.make_fileset([x],'skimmed_llpNanoAOD_v2',location_cfg='backgrounds.yaml',
                                      max_files=max_files_bg,replace_xcache=True,)
         output_data = runner.run(fileset, treename='Events', processor_instance=p)
    except Exception as  e:
         print(f"failed_running {x}, skipping")
         print(e)
         Fail_list.append(x)
         continue
    date_folder = datetime.now().strftime("%Y_%d_%m")
    dir_path = f"OutputFiles/{channel_name}/{date_folder}"
    os.makedirs(dir_path, exist_ok=True)
    filename =  f"{dir_path}/{x}.coffea"
    coffea.util.save(output_data, filename )
    file_names.append(filename)
    print(f"finished running {x}")


if args.run_4mu:
    print("RUNNING 4Mu")
    yaml_file_path = '../../configs/ntuples/signal_4mu_v10.yaml'
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    signals_4mu = list(data["llpNanoAOD_v2"]["samples"].keys())
    signals = [signal for signal in signals_4mu if any(f'{mass}GeV_' in signal for mass in args.masses)]
    fileset = utilities.make_fileset(signals,
                                 "llpNanoAOD_v2",
                                 max_files=max_files_signal, 
                                 location_cfg="signal_4mu_v10.yaml",
                                 replace_xcache=True, )
    output_signal = runner.run(fileset, treename="Events", processor_instance=p)
    date_folder = datetime.now().strftime("%Y_%d_%m")
    dir_path = f"OutputFiles/{channel_name}/{date_folder}"
    os.makedirs(dir_path, exist_ok=True)
    filename =  f"{dir_path}/signal_4mu.coffea"
    coffea.util.save(output_signal, filename )
    file_names.append(filename)
    print("Finished running 4mu")
        
if args.run_2mu:
    print("RUNNING 2Mu")
    yaml_file_path = '../../configs/ntuples/signal_2mu2e_v10.yaml'
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    signals_2mu = list(data["llpNanoAOD_v2"]["samples"].keys())
    signals = [signal for signal in signals_2mu if any(f'{mass}GeV_' in signal for mass in args.masses)]
    fileset = utilities.make_fileset(signals,
                                 "llpNanoAOD_v2",
                                 max_files=max_files_signal, 
                                 location_cfg="signal_2mu2e_v10.yaml",
                                 replace_xcache=True, )
    output_signal = runner.run(fileset, treename="Events", processor_instance=p)
    date_folder = datetime.now().strftime("%Y_%d_%m")
    dir_path = f"OutputFiles/{channel_name}/{date_folder}"
    os.makedirs(dir_path, exist_ok=True)
    filename = f"{dir_path}/signal_2mu.coffea"
    coffea.util.save(output_signal, filename  )
    file_names.append(filename)
    print("Finished running 2mu2e")

client.close()
cluster.close()
print("Here are filenames:")
for x in file_names:
    print(x)
print("Here are the failed files:")
for x in Fail_list:
    print(x)



