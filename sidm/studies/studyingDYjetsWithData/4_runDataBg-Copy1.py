print("IMPORTING MODULES")

import os, sys, subprocess, tempfile
from coffea import processor
import coffea.util
import matplotlib.pyplot as plt
import yaml

sidm_path = str(os.getcwd()).split('/sidm')[0]
if sidm_path not in sys.path:
    sys.path.insert(1, sidm_path)

from sidm.tools import utilities, scaleout, sidm_processor, llpnanoaodschema

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
channel_name ="cr_1LJ_Zmumu_cosmicVeto"
# hist_collection = "deltaR_cosmic"


print("Processor creation")
p = sidm_processor.SidmProcessor(
    [channel_name],
    [  "dyjet_study", ], 
    # unweighted_hist=True,
)

max_files_bg = 1000
max_files_data = -1
max_files_signal = -1


data = [ 
'DoubleMuon_2018C',
]
print("Processing DATA")
Fail_list = []
for x in data:
    print(x)
    try:
        fileset = utilities.make_fileset([x],'llpNanoAOD_v2',location_cfg='data_skimmed.yaml',
                                     max_files=max_files_data,replace_xcache=True,)
        output_data = runner.run(fileset, treename='Events', processor_instance=p)
    except:
        print(f"failed_running {x}, skipping")
        Fail_list.append(x)
        continue
    dir_path = f"OutputFiles/{channel_name}"
    os.makedirs(dir_path, exist_ok=True)
    coffea.util.save(output_data, f"{dir_path}/{x}.coffea"  )
    #REDIR   = "root://cmseos.fnal.gov"
    #EOS_DIR = f"/store/group/lpcmetx/SIDM/coffea_outputs/{os.environ['USER']}/{channel_name}"
    #subprocess.run(["xrdfs", REDIR, "mkdir", "-p", EOS_DIR], check=True)
    #with tempfile.TemporaryDirectory() as tmp:
    #    coffea_local = os.path.join(tmp, f"{x}.coffea")
    #    coffea.util.save(output_data, coffea_local)
    #    print(f"{REDIR}/{EOS_DIR}/{os.path.basename(coffea_local)}")
    #    subprocess.run(["xrdcp", "-f", coffea_local, f"{REDIR}/{EOS_DIR}/{os.path.basename(coffea_local)}"], 
    #                   check=True)
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

print("Processing DATA")
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
    dir_path = f"OutputFiles/{channel_name}"
    os.makedirs(dir_path, exist_ok=True)
    coffea.util.save(output_data, f"{dir_path}/{x}.coffea"  )
    print(f"finished running {x}")
    #REDIR   = "root://cmseos.fnal.gov"
    #EOS_DIR = f"/store/group/lpcmetx/SIDM/coffea_outputs/{os.environ['USER']}/{channel_name}"
    #subprocess.run(["xrdfs", REDIR, "mkdir", "-p", EOS_DIR], check=True)
    #with tempfile.TemporaryDirectory() as tmp:
    #    coffea_local = os.path.join(tmp, f"{x}.coffea")
    #    coffea.util.save(output_data, coffea_local)
    #    print(f"{REDIR}/{EOS_DIR}/{os.path.basename(coffea_local)}")
    #    subprocess.run(["xrdcp", "-f", coffea_local, f"{REDIR}/{EOS_DIR}/{os.path.basename(coffea_local)}"], 
    #                   check=True)
client.close()

cluster.close()
print("Here are the failed files:")
for x in Fail_list:
    print(x)
   


