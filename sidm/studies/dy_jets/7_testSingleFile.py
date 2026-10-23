
import sys
import os
import importlib
# columnar analysis
import awkward as ak
from coffea import processor, nanoevents
# local
sidm_path = str(sys.path[0]).split("/sidm")[0]
if sidm_path not in sys.path: sys.path.insert(1, sidm_path)
from sidm.tools import llpnanoaodschema, sidm_processor,  utilities 
from sidm.studies.cosmicVeto import functions
# always reload local modules to pick up changes during development
importlib.reload(llpnanoaodschema)
importlib.reload(sidm_processor)
#
#
data = functions.get_data_files()
data =  [x for x in data if "C" in x]
DY = functions.get_bg_file_names("DYJetsToLL")


fileset = utilities.make_fileset([DY[1]], "skimmed_llpNanoAOD_v2", location_cfg="backgrounds.yaml", max_files=1,replace_xcache=True, )
file  =  fileset[DY[1]]["files"][0]




events = nanoevents.NanoEventsFactory.from_root(
            {file: "Events"},
                schemaclass=llpnanoaodschema.LLPNanoAODSchema,
                ).events()



muon = events.Muon[
            ak.argsort(events.Muon.pt, axis=1, ascending=False)
            ]
has_two_muons = ak.num(muon) >= 2
events_2mu = events[has_two_muons]
muon_2mu = muon[has_two_muons]
muon0 = muon_2mu[:, 0]  # leading pT
muon1 = muon_2mu[:, 1]  # subleading pT
deltaR = muon0.delta_r(muon1)
mask = deltaR < 0.3
events_selected = events_2mu[mask]
muon_selected = muon_2mu[mask]

genpart = events_selected.GenPart
# genpart_idx = ak.local_index(genpart, axis=1)
gen_muon_mask = (
            (abs(genpart.pdgId) == 13) &
                (genpart.status == 1)
                )

gen_muons = genpart[gen_muon_mask]
# gen_muon_indices = genpart_idx[gen_muon_mask]
sort_idx = ak.argsort(
            gen_muons.pt,
                axis=1,
                    ascending=False
                    )

gen_muons = gen_muons[sort_idx]
# gen_muon_indices = gen_muon_indices[sort_idx]
for x in range(5):
    print(f"\nEvent {x}")
    print(f"{'':10s} {'pT':>20s} {'eta':>20s} {'phi':>20s} {'MotherIdx':>15s}")
    print(f"{'Reco':10s}" f"{str(muon_selected[x].pt):>20s}"f"{str(muon_selected[x].eta):>20s} "
                 f"{str(muon_selected[x].phi):>20s}")
    print(f"{'Gen':10s}  "
            f"{str(gen_muons[x].pt):>20s} "
                    f"{str(gen_muons[x].eta):>20s} "
                            f"{str(gen_muons[x].phi):>20s} "
                                    f"{str(ak.to_list(gen_muons[x].genPartIdxMother)):>15s}")
    

