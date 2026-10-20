import os, sys, subprocess, tempfile
from coffea import processor
import coffea.util
import matplotlib.pyplot as plt
import yaml

sidm_path = str(os.getcwd()).split('/sidm')[0]
if sidm_path not in sys.path:
    sys.path.insert(1, sidm_path)

from sidm.tools import utilities, scaleout, sidm_processor, llpnanoaodschema
utilities.set_plot_style()


os.environ["X509_USER_PROXY"] = "/uscms/home/mjose/x509up_u59701"
scaleout.check_voms_proxy()


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

runner = processor.Runner(
    executor=processor.DaskExecutor(client=client, status=False),
    schema=llpnanoaodschema.LLPNanoAODSchema,
    skipbadfiles=True,
    chunksize=10000,
)


channel_name = "base_ljDisplacementIso"
hist_collection = "signal_highPttest"

p = sidm_processor.SidmProcessor(
    [channel_name],
    [ hist_collection],
    # unweighted_hist=True,
)

max_files_signal = -1


yaml_file_path = '../../configs/ntuples/signal_2mu2e_v10.yaml'
# Open and read the YAML file
with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file)
signals_all = list(data["llpNanoAOD_v2"]["samples"].keys())
masses = [200, 500, 800, 1000]
signals = [
    signal for signal in signals_all
    if any(f'{mass}GeV_' in signal for mass in masses)
]



fileset = utilities.make_fileset(signals,
                                 "llpNanoAOD_v2",
                                 max_files=max_files_signal, 
                                 location_cfg="signal_2mu2e_v10.yaml",
                                 replace_xcache=True, )
output_signal = runner.run(fileset, treename="Events", processor_instance=p)
dir_path = f"OutputFiles/{channel_name}"
os.makedirs(dir_path, exist_ok=True)
coffea.util.save(output_signal, f"{dir_path}/signal_2mu.coffea"  )

# REDIR   = "root://cmseos.fnal.gov"
# EOS_DIR = f"/store/group/lpcmetx/SIDM/coffea_outputs/{os.environ['USER']}/{channel_name}"
# subprocess.run(["xrdfs", REDIR, "mkdir", "-p", EOS_DIR], check=True)
# with tempfile.TemporaryDirectory() as tmp:
#     coffea_local = os.path.join(tmp, "signal_4mu.coffea")
#     coffea.util.save(output_signal, coffea_local)
#     print(f"{REDIR}/{EOS_DIR}/{os.path.basename(coffea_local)}")
#     subprocess.run(["xrdcp", "-f", coffea_local, f"{REDIR}/{EOS_DIR}/{os.path.basename(coffea_local)}"], check=True)

client.close()
cluster.close()
