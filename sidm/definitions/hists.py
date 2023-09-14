"""Define all available histograms

All hists are defined as Histogram objects whose axes are given as a list of Axis objects, which
bundle a hist.axis with a function that defines how the axis will be filled. The underlying
hist.Hists storage is weight unless otherwise specified.
"""

# python
import math
import importlib
# columnar analysis
import hist
import awkward as ak
# local
from sidm.tools import histogram as h
from sidm.tools.utilities import dR, lxy
from sidm.definitions.objects import derived_objs
# always reload local modules to pick up changes during development
importlib.reload(h)


hist_defs = {
    # pv
    "pv_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 100, name="pv_n"),
                   lambda objs, mask: ak.num(objs["pvs"])),
        ],
    ),
    "pv_ndof": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 100, name="pv_ndof"),
                   lambda objs, mask: objs["pvs"].ndof),
        ],
    ),
    "pv_z": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, -50, 50, name="pv_z"),
                   lambda objs, mask: objs["pvs"].z),
        ],
    ),
    "pv_rho": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, -0.5, 0.5, name="pv_rho"),
                   lambda objs, mask: objs["pvs"].rho),
        ],
    ),
    # pfelectron
    "electron_n": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="electron_n"),
                   lambda objs, mask: ak.num(objs["electrons"])),
        ],
    ),
    "electron_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="electron_pt"),
                   lambda objs, mask: objs["electrons"].pt),
        ],
    ),
    "electron_eta_phi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="electron_eta"),
                   lambda objs, mask: objs["electrons"].eta),
            h.Axis(hist.axis.Regular(50, -1*math.pi, math.pi, name="electron_phi"),
                   lambda objs, mask: objs["electrons"].phi),
        ],
    ),
    # pfphoton
    "photon_n": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="photon_n"),
                   lambda objs, mask: ak.num(objs["photons"])),
        ],
    ),
    "photon_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="photon_pt"),
                   lambda objs, mask: objs["photons"].pt),
        ],
    ),
    "photon_eta_phi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="photon_eta"),
                   lambda objs, mask: objs["photons"].eta),
            h.Axis(hist.axis.Regular(50, -1*math.pi, math.pi, name="photon_phi"),
                   lambda objs, mask: objs["photons"].phi),
        ],
    ),
    # pfmuon
    "muon_n": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="muon_n"),
                   lambda objs, mask: ak.num(objs["muons"])),
        ],
    ),
    "muon_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="muon_pt"),
                   lambda objs, mask: objs["muons"].pt),
        ],
    ),
    "muon_eta_phi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="muon_eta"),
                   lambda objs, mask: objs["muons"].eta),
            h.Axis(hist.axis.Regular(50, -1*math.pi, math.pi, name="muon_phi"),
                   lambda objs, mask: objs["muons"].phi),
        ],
    ),
    "muon_absD0": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name="muon_absD0", label=r"Muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["muons"].d0)),
        ],
    ),
    "muon_absD0_lowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 10, name="muon_absD0_lowRange",
                                     label=r"Muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["muons"].d0)),
        ],
    ),
    # dsamuon
    "dsaMuon_n": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="dsaMuon_n"),
                   lambda objs, mask: ak.num(objs["dsaMuons"])),
        ],
    ),
    "dsaMuon_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="dsaMuon_pt"),
                   lambda objs, mask: objs["dsaMuons"].pt),
        ],
    ),
    "dsaMuon_eta_phi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="dsaMuon_eta"),
                   lambda objs, mask: objs["dsaMuons"].eta),
            h.Axis(hist.axis.Regular(50, -1*math.pi, math.pi, name="dsaMuon_phi"),
                   lambda objs, mask: objs["dsaMuons"].phi),
        ],
    ),
    "dsaMuon_absD0": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name="dsaMuon_absD0", label=r"Muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"].d0)),
        ],
    ),
    "dsaMuon_absD0_lowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 10, name="dsaMuon_absD0_lowRange",
                                     label=r"Muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"].d0)),
        ],
    ),
    # lj
    "lj_n": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="lj_n"),
                   lambda objs, mask: ak.num(objs["ljs"])),
        ],
    ),
    "lj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 100, name="lj_pt", label="Lepton jet pT [GeV]"),
                   lambda objs, mask: objs["ljs"].pt),
        ],
    ),
    "lj_pfIsolation05": h.Histogram(
        [
            h.Axis(hist.axis.Regular(80, 0, 0.8, name="lj_pfIsolation05",
                                     label="Lepton jet isolation"),
                   lambda objs, mask: objs["ljs"].pfIsolation05),
        ],
    ),
    "lj0_pfIsolation05": h.Histogram(
        [
            h.Axis(hist.axis.Regular(80, 0, 0.8, name="lj_pfIsolation05",
                                     label="Leading lepton jet isolation"),
                   lambda objs, mask: objs["ljs"][mask, 0].pfIsolation05),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 0,
    ),
    "lj1_pfIsolation05": h.Histogram(
        [
            h.Axis(hist.axis.Regular(80, 0, 0.8, name="lj_pfIsolation05",
                                     label="Subleading lepton jet isolation"),
                   lambda objs, mask: objs["ljs"][mask, 1].pfIsolation05),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_pfIsolationPtNoPU05": h.Histogram(
        [
            h.Axis(hist.axis.Regular(80, 0, 0.8, name="lj_pfIsolationPtNoPU05",
                                     label="Lepton jet isolation"),
                   lambda objs, mask: objs["ljs"].pfIsolationPtNoPU05),
        ],
    ),
    "lj_pfIsolationPt05": h.Histogram(
        [
            h.Axis(hist.axis.Regular(80, 0, 0.8, name="lj_pfIsolationPt05",
                                     label="Lepton jet isolation"),
                   lambda objs, mask: objs["ljs"].pfIsolationPt05),
        ],
    ),
    "lj_pfIsolation07": h.Histogram(
        [
            h.Axis(hist.axis.Regular(80, 0, 0.8, name="lj_pfIsolation07",
                                     label="Lepton jet isolation"),
                   lambda objs, mask: objs["ljs"].pfIsolation07),
        ],
    ),
    "lj_pfIsolationPtNoPU07": h.Histogram(
        [
            h.Axis(hist.axis.Regular(80, 0, 0.8, name="lj_pfIsolationPtNoPU07",
                                     label="Lepton jet isolation"),
                   lambda objs, mask: objs["ljs"].pfIsolationPtNoPU07),
        ],
    ),
    "lj_pfIsolationPt07": h.Histogram(
        [
            h.Axis(hist.axis.Regular(80, 0, 0.8, name="lj_pfIsolationPt07",
                                     label="Lepton jet isolation"),
                   lambda objs, mask: objs["ljs"].pfIsolationPt07),
        ],
    ),
    "lj_pfiso": h.Histogram(
        [
            h.Axis(hist.axis.Regular(80, 0, 0.8, name="lj_pfiso",
                                     label="Lepton jet isolation"),
                   lambda objs, mask: objs["ljs"].pfiso),
        ],
    ),
    "lj0_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 100, name="lj0_pt",
                                     label="Leading lepton jet pT [GeV]"),
                   lambda objs, mask: objs["ljs"][mask, 0].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 0,
    ),
    "lj1_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 100, name="lj1_pt",
                                     label="Subleading lepton jet pT [GeV]"),
                   lambda objs, mask: objs["ljs"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj0_e": h.Histogram(
        [
            h.Axis(hist.axis.Regular(350, 0, 700, name="lj_e",
                                     label="Leading lepton jet E [GeV]"),
                   lambda objs, mask: objs["ljs"][mask, 0].energy),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 0,
    ),
    "lj1_e": h.Histogram(
        [
            h.Axis(hist.axis.Regular(350, 0, 700, name="lj_e",
                                     label="Subleading lepton jet E [GeV]"),
                   lambda objs, mask: objs["ljs"][mask, 1].energy),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj0_dRSpread": h.Histogram(
        [
            h.Axis(hist.axis.Regular(350, 0, 0.05, name="lj0_dRSpread",
                                     label="Leading lepton jet dRSpread"),
                   lambda objs, mask: objs["ljs"][mask, 0].dRSpread),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 0,
    ),
    "lj1_dRSpread": h.Histogram(
        [
            h.Axis(hist.axis.Regular(350, 0, 0.05, name="lj1_dRSpread",
                                     label="Subleading lepton jet dRSpread"),
                   lambda objs, mask: objs["ljs"][mask, 1].dRSpread),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_eta_phi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="lj_eta"),
                   lambda objs, mask: objs["ljs"].eta),
            h.Axis(hist.axis.Regular(50, -1*math.pi, math.pi, name="lj_phi"),
                   lambda objs, mask: objs["ljs"].phi),
        ],
    ),
    "egm_lj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 100, name="egm_lj_pt",
                                     label="EGM-type lepton jet pT [GeV]"),
                   lambda objs, mask: derived_objs["egm_ljs"](objs).pt),
        ],
    ),
    "mu_lj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 100, name="mu_lj_pt",
                                     label="Mu-type lepton jet pT [GeV]"),
                   lambda objs, mask: derived_objs["mu_ljs"](objs).pt),
        ],
    ),
    "lj_electronN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="lj_electronN"),
                   lambda objs, mask: objs["ljs"].electron_n),
        ],
    ),
    "lj_photonN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="lj_photonN"),
                   lambda objs, mask: objs["ljs"].photon_n),
        ],
    ),
    "lj_electronPhotonN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="lj_electronPhotonN"),
                   lambda objs, mask: objs["ljs"].electron_n + objs["ljs"].photon_n),
        ],
    ),
    "lj_muonN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="lj_muonN"),
                   lambda objs, mask: objs["ljs"].muon_n),
        ],
    ),
    # ljsource
    "ljsource_n": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="ljsource_n"),
                   lambda objs, mask: ak.num(objs["ljsources"])),
        ],
    ),
    "ljsource_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 100, name="ljsource_pt",
                                     label="Lepton jet source pT [GeV]"),
                   lambda objs, mask: objs["ljsources"].pt),
        ],
    ),
    "ljsource_eta_phi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="ljsource_eta"),
                   lambda objs, mask: objs["ljsources"].eta),
            h.Axis(hist.axis.Regular(50, -1*math.pi, math.pi, name="ljsource_phi"),
                   lambda objs, mask: objs["ljsources"].phi),
        ],
    ),
    "ljsource_charge": h.Histogram(
        [
            h.Axis(hist.axis.Integer(-1, 1, name="ljsource_charge"),
                   lambda objs, mask: objs["ljsources"].charge),
        ],
    ),
    "ljsource_type": h.Histogram(
        [
            h.Axis(hist.axis.IntCategory([2, 3, 4, 8], name="lj_type"),
                   lambda objs, mask: objs["ljsources"]["type"]), # avoid ak.Array.type
        ],
    ),
    # pfelectron-lj
    "electron_lj_dR": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="electron_lj_dR"),
                   lambda objs, mask: dR(objs["electrons"], objs["ljs"]))
        ],
    ),
    "electron_lj_dR_lowRange": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 1.0, name="electron_lj_dR_lowRange"),
                   lambda objs, mask: dR(objs["electrons"], objs["ljs"]))
        ],
    ),
    # pfphoton-lj
    "photon_lj_dR": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="photon_lj_dR"),
                   lambda objs, mask: dR(objs["photons"], objs["ljs"]))
        ],
    ),
    "photon_lj_dR_lowRange": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 1.0, name="photon_lj_dR_lowRange"),
                   lambda objs, mask: dR(objs["photons"], objs["ljs"]))
        ],
    ),
    "photon_lj_dR_reallyLowRange": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 0.1, name="photon_lj_dR_reallyLowRange"),
                   lambda objs, mask: dR(objs["photons"], objs["ljs"]))
        ],
    ),
    # pfmuon-lj
    "muon_lj_dR": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="muon_lj_dR"),
                   lambda objs, mask: dR(objs["muons"], objs["ljs"]))
        ],
    ),
    "muon_lj_dR_lowRange": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 1.0, name="muon_lj_dR_lowRange"),
                   lambda objs, mask: dR(objs["muons"], objs["ljs"]))
        ],
    ),
    # dsamuon-lj
    "dsaMuon_lj_dR": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="dsaMuon_lj_dR"),
                   lambda objs, mask: dR(objs["dsaMuons"], objs["ljs"]))
        ],
    ),
    "dsaMuon_lj_dR_lowRange": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 1.0, name="dsaMuon_lj_dR_lowRange"),
                   lambda objs, mask: dR(objs["dsaMuons"], objs["ljs"]))
        ],
    ),
    # lj-lj
    "lj_lj_absdphi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="ljlj_absdphi"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].phi
                                          - objs["ljs"][mask, 0].phi)),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_invmass": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 2000, name="ljlj_mass"),
                   lambda objs, mask: objs["ljs"][mask, :2].sum().mass),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_invmass_lowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name="ljlj_mass"),
                   lambda objs, mask: objs["ljs"][mask, :2].sum().mass),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    # ABCD plane
    "abcd_lj_lj_dphi_vs_lj0_pfIsolationPt05": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2*math.pi, name="ljlj_absdphi",
                                     label=fr"Lepton jet pair |$\Delta\phi$|"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].phi
                                          - objs["ljs"][mask, 0].phi)),
            h.Axis(hist.axis.Regular(80, 0, 0.8, name="lj_pfIsolationPt05",
                                     label="Leading lepton jet isolation"),
                   lambda objs, mask: objs["ljs"][mask, 0].pfIsolationPt05),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    # gen
    "gen_abspid": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 40, name="gen_abspid"),
                   lambda objs, mask: abs(objs["gens"].pid)),
        ],
    ),
    # genelectron
    "genE_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genE_pt"),
                   lambda objs, mask: abs(objs["genEs"].pt)),
        ],
    ),
    "genE_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 700, name="genE_pt"),
                   lambda objs, mask: abs(objs["genEs"].pt)),
        ],
    ),
    "genE0_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genE0_pt_lowRange"),
                   lambda objs, mask: objs["genEs"][mask, 0].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 0,
    ),
    "genE1_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genE1_pt_lowRange"),
                   lambda objs, mask: objs["genEs"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genE0_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 1000, name="genE0_pt"),
                   lambda objs, mask: objs["genEs"][mask, 0].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 0,
    ),
    "genE1_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 1000, name="genE1_pt"),
                   lambda objs, mask: objs["genEs"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    # genelectron-genelectron
    "genE_genE_dR": h.Histogram(
        [
            # dR(subleading gen E, leading gen E)
            h.Axis(hist.axis.Regular(50, 0, 1.0, name="genE_genE_dR"),
                   lambda objs, mask: objs["genEs"][mask, 1].delta_r(objs["genEs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genE_genE_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genE_genE_pt"),
                   lambda objs, mask: objs["genEs"][mask, :2].sum().pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    # genmuon
    "genMu_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genMu_pt"),
                   lambda objs, mask: abs(objs["genMus"].pt)),
        ],
    ),
    "genMu_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 700, name="genMu_pt"),
                   lambda objs, mask: abs(objs["genMus"].pt)),
        ],
    ),
    "genMu0_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 100, name="genMu0_pt"),
                   lambda objs, mask: objs["genMus"][mask, 0].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 0
    ),
    "genMu1_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 100, name="genMu1_pt"),
                   lambda objs, mask: objs["genMus"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu0_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 1000, name="genMu0_pt"),
                   lambda objs, mask: objs["genMus"][mask, 0].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 0,
    ),
    "genMu1_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 1000, name="genMu1_pt"),
                   lambda objs, mask: objs["genMus"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    # genmuon-genmuon
    "genMu_genMu_dR": h.Histogram(
        [
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(50, 0, 1.0, name="genMu_genMu_dR"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(
                       objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu_genMu_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genMu_genMu_pt"),
                   lambda objs, mask: objs["genMus"][mask, :2].sum().pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    # gen dark photons (A)
    "genA_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(objs["genAs"].pt)),
        ],
    ),
    "genA_eta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="genA_eta"),
                   lambda objs, mask: objs["genAs"].eta),
        ],
    ),
    "genA_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="genA_n"),
                   lambda objs, mask: ak.num(objs["genAs"])),
        ],
    ),
    "genA_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(objs["genAs"]) ),
        ],
    ),
    "genAs_toMu_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(objs["genAs_toMu"]) ),
        ],
    ),
    "genAs_toMu_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(objs["genAs_toMu"].pt) ),
        ],
    ),
    "genAs_toMu_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(140, 0, 700, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(objs["genAs_toMu"].pt) ),
        ],
    ),
    "genAs_toMu_eta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="genAs_toMu_eta"),
                   lambda objs, mask: objs["genAs_toMu"].eta ),
        ],
    ),
    "genAs_toE_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(objs["genAs_toE"]) ),
        ],
    ),
    "genAs_toE_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(objs["genAs_toE"].pt) ),
        ],
    ),
    "genAs_toE_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(140, 0, 700, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(objs["genAs_toE"].pt) ),
        ],
    ),
    "genAs_toE_eta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="genAs_toE_eta"),
                   lambda objs, mask: objs["genAs_toE"].eta ),
        ],
    ),
    "genAs_toMu_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="genAs_toMu_n"),
                   lambda objs, mask: ak.num(objs["genAs_toMu"]) ),
        ],
    ),
    "genA_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(140, 0, 700, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(objs["genAs"].pt)),
        ],
    ),
    "genA_eta_phi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="genA_eta"),
                   lambda objs, mask: objs["genAs"].eta),
            h.Axis(hist.axis.Regular(50, -1*math.pi, math.pi, name="genA_phi"),
                   lambda objs, mask: objs["genAs"].phi),
        ],
    ),
    "genA_pt_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genA_pt"),
                   lambda objs, mask: abs(objs["genAs"].pt)),
            h.Axis(hist.axis.Regular(250, 0, 500, name="genA_lxy"),
                   lambda objs, mask: lxy(objs["genAs"])),
        ],
    ),
    # genA-genA
    "genA_genA_dphi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="genA_genA_dphi"),
                   lambda objs, mask: abs(objs["genAs"][mask, 1].phi
                                          - objs["genAs"][mask, 0].phi)),
        ],
        evt_mask=lambda objs: ak.num(objs["genAs"]) > 1,
    ),
    # genA-LJ
    "genA_lj_dR": h.Histogram(
        [
            # dR(A, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="genA_lj_dR"),
                   lambda objs, mask: dR(objs["genAs"], objs["ljs"]))
        ],
    ),
    "genA_lj_dR_lowRange": h.Histogram(
        [
            # dR(A, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 1.0, name="genA_lj_dR_lowRange"),
                   lambda objs, mask: dR(objs["genAs"], objs["ljs"]))
        ],
    ),
    "lj_genA_ptRatio": h.Histogram(
        [
            # (LJ pT)/(nearest A pT)
            h.Axis(hist.axis.Regular(50, 0, 2.0, name="lj_genA_ptRatio"),
                   lambda objs, mask: objs["ljs"].pt
                       / objs["ljs"].nearest(objs["genAs"]).pt),
        ],
    ),
    "egm_lj_genA_ptRatio": h.Histogram(
        [
            # (LJ pT)/(nearest A pT)
            h.Axis(hist.axis.Regular(50, 0, 2.0, name="egm_lj_genA_ptRatio"),
                   lambda objs, mask: derived_objs["egm_ljs"](objs).pt
                       / derived_objs["egm_ljs"](objs).nearest(objs["genAs"]).pt),
        ],
    ),
    "mu_lj_genA_ptRatio": h.Histogram(
        [
            # (LJ pT)/(nearest A pT)
            h.Axis(hist.axis.Regular(50, 0, 2.0, name="mu_lj_genA_ptRatio"),
                   lambda objs, mask: derived_objs["mu_ljs"](objs).pt
                       / derived_objs["mu_ljs"](objs).nearest(objs["genAs"]).pt),
        ],
    ),
    "genA_matched_lj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name="matched_genA_lxy"),
                   lambda objs, mask: lxy(derived_objs["genAs_matched_lj"](objs, 0.4)) ),
        ],
    ),
    "genA_matched_lj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_matched_lj"](objs, 0.4).pt) ),
        ],
    ),
    "genA_matched_lj_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(140, 0, 700, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_matched_lj"](objs, 0.4).pt) ),
        ],
    ),
    "genA_matched_lj_eta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="genA_matched_lj_eta"),
                   lambda objs, mask: derived_objs["genAs_matched_lj"](objs, 0.4).eta ),
        ],
    ),
    "genA_toMu_matched_lj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_toMu_matched_lj"](objs, 0.4)) ),
        ],
    ),
    "genA_toE_matched_lj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_toE_matched_lj"](objs, 0.4)) ),
        ],
    ),
    "genA_matched_muLj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_matched_muLj"](objs, 0.4)) ),
        ],
    ),
    "genA_toMu_matched_muLj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_toMu_matched_muLj"](objs, 0.4)) ),
        ],
    ),
    "genA_toMu_matched_muLj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_toMu_matched_muLj"](objs, 0.4).pt) ),
        ],
    ),
    "genA_toMu_matched_muLj_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(140, 0, 700, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_toMu_matched_muLj"](objs, 0.4).pt) ),
        ],
    ),
    "genA_toMu_matched_muLj_eta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="genA_toMu_matched_muLj_eta"),
                   lambda objs, mask: derived_objs["genAs_toMu_matched_muLj"](objs, 0.4).eta ),
        ],
    ),
    "genA_matched_egmLj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_matched_egmLj"](objs, 0.4)) ),
        ],
    ),
    "genA_toE_matched_egmLj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4)) ),
        ],
    ),
    "genA_toE_matched_egmLj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4).pt) ),
        ],
    ),
    "genA_toE_matched_egmLj_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(140, 0, 700, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4).pt) ),
        ],
    ),
    "genA_toE_matched_egmLj_eta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name="genA_toE_matched_egmLj_eta"),
                   lambda objs, mask: derived_objs["genAs_toE_matched_egmLj"](objs, 0.4).eta ),
        ],
    ),
    "genA_matched_lj_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="genA_matched_n"),
                   lambda objs, mask: ak.num(derived_objs["genAs_matched_lj"](objs, 0.4)) ),
        ],
    ),
}
