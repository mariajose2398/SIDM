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
channel_name = "cr_1egmOr1mu_lj_withoutEta_phi_cosmic_veto_disp"
# hist_collection = "deltaR_cosmic"


print("Processor creation")
p = sidm_processor.SidmProcessor(
    [channel_name],
    [  "cosmic_veto", ], 
    # unweighted_hist=True,
)

max_files_bg = -1
max_files_data = -1
max_files_signal = -1


data = [ 
 'DoubleMuon_2018C',
 'DoubleMuon_2018A_0',
 'DoubleMuon_2018A_1',
 'DoubleMuon_2018A_2',
 'DoubleMuon_2018B_0',
 'DoubleMuon_2018B_1',
 'DoubleMuon_2018B_2',
 'DoubleMuon_2018B_3',
 'DoubleMuon_2018B_4',
 'DoubleMuon_2018B_5',
 'DoubleMuon_2018B_6',
 'DoubleMuon_2018B_7',
 'DoubleMuon_2018B_8',
 'DoubleMuon_2018B_9',
 'DoubleMuon_2018B_10',
 'DoubleMuon_2018B_11',
 'DoubleMuon_2018B_12',
 'DoubleMuon_2018B_13',
 'DoubleMuon_2018D_0',
 'DoubleMuon_2018D_1',
 'DoubleMuon_2018D_10',
 'DoubleMuon_2018D_11',
 'DoubleMuon_2018D_12',
 'DoubleMuon_2018D_13',
 'DoubleMuon_2018D_14',
 'DoubleMuon_2018D_15',
 'DoubleMuon_2018D_16',
 'DoubleMuon_2018D_17',
 'DoubleMuon_2018D_18',
 'DoubleMuon_2018D_19',
 'DoubleMuon_2018D_2',
 'DoubleMuon_2018D_20',
 'DoubleMuon_2018D_21',
 'DoubleMuon_2018D_22',
 'DoubleMuon_2018D_23',
 'DoubleMuon_2018D_24',
 'DoubleMuon_2018D_25',
 'DoubleMuon_2018D_26',
 'DoubleMuon_2018D_27',
 'DoubleMuon_2018D_28',
 'DoubleMuon_2018D_29',
 'DoubleMuon_2018D_3', #error
 'DoubleMuon_2018D_30',
 'DoubleMuon_2018D_31',
 'DoubleMuon_2018D_32',
 'DoubleMuon_2018D_33',
 'DoubleMuon_2018D_34',
 'DoubleMuon_2018D_35',
 'DoubleMuon_2018D_36',
'DoubleMuon_2018D_37',
'DoubleMuon_2018D_38',
 'DoubleMuon_2018D_39',
 'DoubleMuon_2018D_4',
 'DoubleMuon_2018D_40',
 'DoubleMuon_2018D_41',
 'DoubleMuon_2018D_42',
 'DoubleMuon_2018D_43',
 'DoubleMuon_2018D_44',
 'DoubleMuon_2018D_45',
 'DoubleMuon_2018D_46',
 'DoubleMuon_2018D_47',
 'DoubleMuon_2018D_48',
 'DoubleMuon_2018D_49',
 'DoubleMuon_2018D_5',
 'DoubleMuon_2018D_6',
 'DoubleMuon_2018D_7',
 'DoubleMuon_2018D_8',
 'DoubleMuon_2018D_9',
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
    REDIR   = "root://cmseos.fnal.gov"
    EOS_DIR = f"/store/group/lpcmetx/SIDM/coffea_outputs/{os.environ['USER']}/{channel_name}"
    subprocess.run(["xrdfs", REDIR, "mkdir", "-p", EOS_DIR], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        coffea_local = os.path.join(tmp, f"{x}.coffea")
        coffea.util.save(output_data, coffea_local)
        print(f"{REDIR}/{EOS_DIR}/{os.path.basename(coffea_local)}")
        subprocess.run(["xrdcp", "-f", coffea_local, f"{REDIR}/{EOS_DIR}/{os.path.basename(coffea_local)}"], 
                       check=True)
client.close()

cluster.close()
print("Here are the failed files:")
for x in Fail_list:
    print(x)
   


