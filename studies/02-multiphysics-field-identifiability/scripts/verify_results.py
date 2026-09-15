"""Check every number quoted in RESULTS.md against the files it cites.

    python scripts/verify_results.py [--results results]

This exists so that a reader does not have to take the prose on trust, and so
that a change to the code that moves a reported number fails loudly rather than
silently leaving the document wrong.  It reads only the machine-readable
outputs; it does not re-run anything.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys

FAILURES = []
CHECKS = 0


def chk(label, got, want, tol=6e-4, absolute=True):
    global CHECKS
    CHECKS += 1
    g, w = float(got), float(want)
    d = abs(g - w) if absolute else abs(g - w) / max(abs(w), 1e-300)
    if d > tol:
        FAILURES.append((label, g, w, d))


def load(root, name):
    with open(os.path.join(root, name)) as fh:
        return json.load(fh)


def table(root, name, field):
    out = {}
    with open(os.path.join(root, name)) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["config"], {})[r["quantity"]] = float(r[field])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = a.results

    # ---- section 1 -------------------------------------------------------
    g = load(R, "E0_screening.json")
    for k, v in dict(loss_tangent=0.628, biot_frequency_ratio=4.76e-3,
                     biot_inertial_coupling=1.34e-4, shear_wavelength_mm=9.56,
                     aperture_in_wavelengths=1.05, aperture_attenuation=1.89,
                     peclet=6.26, transit_fraction=15.7, storage_ratio=2.69,
                     relaxing_fraction=0.271,
                     consolidation_coefficient_m2_s=2.60e-6,
                     consolidation_time_s=3.89).items():
        chk("group " + k, g["groups"][k], v, 0.01, absolute=False)
    for k, v in dict(rho=49.3, alpha=6.6, eta_b=2.5).items():
        chk("fixed " + k, g["fixed_screening"][k], v, 0.01, absolute=False)

    # ---- section 2 -------------------------------------------------------
    e1 = load(R, "E1_swe.json")
    assert e1["null_labels_log"] == ["S_v^+1.00"], e1["null_labels_log"]
    chk("E1 lam phi", e1["eigenvalues"][1], 3.30e-8, 0.01, absolute=False)
    chk("E1 lam k", e1["eigenvalues"][2], 1.46e-3, 0.01, absolute=False)
    chk("E1 lam Tp", e1["eigenvalues"][3], 1.40, 0.005, absolute=False)
    chk("E1 contr Tp", e1["contraction"][3], 0.645, 0.005, absolute=False)
    chk("E1 block lo", e1["eigenvalues"][4], 2.80e4, 0.005, absolute=False)
    chk("E1 block hi", e1["eigenvalues"][6], 5.55e5, 0.005, absolute=False)
    chk("E1 sens k", e1["sensitivity"]["k"], 0.11, 0.05, absolute=False)
    chk("E1 sens phi", e1["sensitivity"]["phi"], 3e-4, 0.1, absolute=False)

    e2 = load(R, "E2_ceus.json")
    assert "phi^+0.71 A^-0.71" in e2["null_labels_log"]
    for q, v in dict(D=0.012, vmag=0.003, t0=0.005, phi=0.838, A=0.545).items():
        chk("E2 " + q, e2["summary"][q]["contraction"], v)
    chk("E2cal phi", load(R, "E2_ceus_Acal.json")["summary"]["phi"]["contraction"],
        0.066)

    # ---- section 3 -------------------------------------------------------
    e2n = load(R, "E2n_ceus_norm.json")
    assert e2n["rank"] == 3, e2n["rank"]
    s = e2n["sensitivity"]
    mx = max(s.values())
    for q in ("phi", "A", "vth", "vaz"):
        assert s[q] < 1e-12 * mx, (q, s[q] / mx)

    # ---- sections 4 to 6 -------------------------------------------------
    con = table(R, "summary_table.csv", "contraction")
    expect = {
        "E3_joint_independent": dict(mu=0.003, eta_s=0.006, phi=0.066, D=0.011,
                                     vmag=0.003, k=0.999, S_v=1.000),
        "E4_joint_constitutive": dict(k=0.999, S_v=0.839, phi=0.066),
        "E4_joint_constitutive_Cvcal": dict(k=0.999, S_v=0.269, phi=0.066),
        "E5_joint_network": dict(k=0.486, S_v=0.839, phi=0.066),
        "E5_ceus_only_network": dict(k=0.486, S_v=0.839, phi=0.066),
        "E8_relax_independent": dict(k=0.352, S_v=0.441, M=0.350, phi=0.066),
        "E8_relax_independent_matched": dict(k=0.298, S_v=0.656, M=0.295, phi=0.066),
        "E8_relax_constitutive": dict(k=0.339, S_v=0.424, M=0.337, phi=0.066),
        "E8_relax_constitutive_Cvcal": dict(k=0.132, S_v=0.257, M=0.118, phi=0.066),
        "E8_relax_network": dict(k=0.290, S_v=0.365, M=0.290, phi=0.065),
        "E8_relax_network_Cvcal": dict(k=0.129, S_v=0.253, M=0.116, phi=0.065),
        "E8_relax_constitutive_Cvcal_norm": dict(k=0.298, S_v=0.656, M=0.295,
                                                 phi=0.685),
        "E8_relax_constitutive_Pcal": dict(k=0.046, S_v=0.059, M=0.039, phi=0.066),
    }
    # the shear window adds nothing to the network route
    chk("network route: joint equals contrast alone",
        con["E5_joint_network"]["k"], con["E5_ceus_only_network"]["k"], 1e-3)
    for c, qs in expect.items():
        for q, v in qs.items():
            chk("%s %s" % (c, q), con[c][q], v)

    e8 = load(R, "E8_relax_independent.json")
    assert e8["rank"] == e8["n_par"] - 2, (e8["rank"], e8["n_par"])
    assert set(e8["null_labels_log"]) == {
        "phi^-0.71 A^+0.71", "M^-0.63 P^-0.32 L^-0.32 S_v^+0.63"}, \
        e8["null_labels_log"]
    assert e8["eig_labels_log"][2] == "k^+0.50 M^-0.50 P^-0.50 S_v^+0.50", \
        e8["eig_labels_log"][2]
    chk("the k direction is nearly null", e8["eigenvalues"][2], 1.8e-4, 0.05,
        absolute=False)
    e8c = load(R, "E8_relax_constitutive_Cvcal.json")
    assert e8c["null_labels_log"][0].startswith("phi^-0.61 A^+0.61"), \
        e8c["null_labels_log"]

    # ---- sections 6 and 7: profiles at three levels ---------------------
    pr = load(R, "profiles.json")
    for c, lv, v in [("E3_joint_independent", "imaging", 0.69),
                     ("E5_joint_network", "imaging", 0.69),
                     ("E8_relax_independent_matched", "imaging", 0.69),
                     ("E8_relax_constitutive_Cvcal", "imaging", 0.69),
                     ("E3_joint_independent", "imaging_plus_auxiliary", 7.2),
                     ("E8_relax_independent_matched", "imaging_plus_auxiliary", 349.9),
                     ("E8_relax_constitutive_Cvcal", "imaging_plus_auxiliary", 570.3),
                     ("E3_joint_independent", "posterior", 15.1),
                     ("E8_relax_independent_matched", "posterior", 388.7),
                     ("E8_relax_constitutive_Cvcal", "posterior", 609.1)]:
        chk("profile %s %s" % (c, lv), max(pr[c]["k"]["delta_" + lv]), v,
            0.02, absolute=False)
    # the imaging likelihood is the same in every configuration
    imaging = [max(pr[c]["k"]["delta_imaging"]) for c in pr]
    chk("imaging profile is configuration independent",
        max(imaging) - min(imaging), 0.0, 0.02)

    # ---- what the profile optimizer did ---------------------------------
    # a flat profile is a statement about the likelihood only if the minimizer
    # converged and stayed inside the physical domain
    for c in pr:
        for lv in ("imaging", "imaging_plus_auxiliary", "posterior"):
            au = pr[c]["k"]["audit_" + lv]
            assert au["n_not_converged"] == 0, (c, lv, au["n_not_converged"])
            for q, v in au["per_parameter"].items():
                assert v["outside_domain"] == 0, (c, lv, q)
        # the imaging profile reaches the ceiling of the volume fraction
        assert "phi" in pr[c]["k"]["audit_imaging"]["parameters_on_a_bound"], c
        chk("porosity ceiling reached, %s" % c,
            pr[c]["k"]["audit_imaging"]["per_parameter"]["phi"]["max"], 1.0,
            1e-6, absolute=False)
    # and the flatness does not depend on it
    for c in ("E3_joint_independent", "E8_relax_independent_matched",
              "E8_relax_constitutive_Cvcal"):
        ds = pr[c]["k"]["domain_sensitivity_imaging"]
        assert ds["ceiling"] == {"phi": 0.30}, ds["ceiling"]
        chk("domain check unchanged, %s" % c, ds["max_abs_difference"], 0.0,
            0.01)
        chk("domain check maximum, %s" % c, ds["delta_max_tightened"],
            ds["delta_max"], 0.01)

    # ---- sampling validation of the Gaussian approximation --------------
    mc = load(R, "mcmc_validation.json")
    assert mc["missing"] == [], mc["missing"]
    for r in mc["configurations"]:
        assert r["worst_rhat"] < 1.02, (r["name"], r["worst_rhat"])
        assert r["worst_rhat_direction"] < 1.02, r["name"]
        assert r["min_ess"] > 300, (r["name"], r["min_ess"])
        for q, v in r["comparison"].items():
            assert 0.9 < v["width_ratio_gaussian_over_sampled"] < 1.1, \
                (r["name"], q, v["width_ratio_gaussian_over_sampled"])
            assert abs(v["location_shift_in_sampled_sd"]) < 0.25, (r["name"], q)
    ef = mc["effect"]
    chk("sampled coupling ratio", ef["sampled"]["ratio"], 2.25, 0.02,
        absolute=False)
    chk("approximated coupling ratio", ef["gaussian"]["ratio"], 2.24, 0.02,
        absolute=False)
    chk("sampled k, matched independent", ef["sampled"]["independent"], 0.295,
        0.02, absolute=False)
    chk("sampled k, coupled", ef["sampled"]["coupled"], 0.131, 0.02,
        absolute=False)
    chk("normalizing returns the sampled k to the control",
        mc["normalized_control"]["ratio"], 1.01, 0.02, absolute=False)

    np_ = load(R, "network_prior.json")
    chk("prior corr phi k", np_["corr"][0][1], 0.873, 0.005, absolute=False)
    chk("prior corr phi D", np_["corr"][0][2], -0.325, 0.01, absolute=False)
    chk("prior cond sd k", np_["conditional_sd_given_phi"]["k"], 0.487, 0.005,
        absolute=False)
    assert (np_["n_ensemble"], np_["n_rows_raw"], np_["n_censored_dropped"],
            np_["n_seeds"]) == (30, 40, 10, 6)
    chk("E5 k equals prior conditional spread", con["E5_joint_network"]["k"],
        np_["conditional_sd_given_phi"]["k"], 0.002)

    # ---- the near-invariant direction -----------------------------------
    # exact for the relaxation and contrast windows, not for the shear one
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import numpy as _np
    from fieldid import Acquisition, Inversion
    acq = Acquisition(); acq.relax.enabled = True
    inv = Inversion(free=("mu",), windows=("swe", "ceus", "relax"), acq=acq)
    p0 = inv.truth_physical()
    q = dict(p0); a_ = 10.0
    q["k"] *= a_; q["C_v"] *= a_; q["S_v"] = q["phi"] * q["C_v"]
    q["M"] /= a_; q["P"] /= a_
    sl, sig = inv.window_slices(), inv.sigma()
    d = {w: float(((( inv._window(w, q) - inv._window(w, p0)) / sig[sl[w]]) ** 2).sum())
         for w in ("relax", "ceus", "swe")}
    chk("near-invariant direction, relaxation", d["relax"], 0.0, 1e-12)
    chk("near-invariant direction, contrast", d["ceus"], 0.0, 1e-12)
    chk("near-invariant direction, shear", d["swe"], 0.238, 0.02, absolute=False)

    # ---- section 8 -------------------------------------------------------
    bias = table(R, "discrepancy_table.csv", "bias_log")
    for c, q, v in [("E7_E1_swe_push", "eta_s", -0.424),
                    ("E7_E1_swe_push", "mu", -0.050),
                    ("E7_E1_swe_push", "phi", 2.947),
                    ("E7_E1_swe_push", "k", 7.577),
                    ("E7_E2_ceus_Acal_twocompartment", "phi", -0.122),
                    ("E7_E2_ceus_Acal_twocompartment", "D", 0.066),
                    ("E7_E2_ceus_Acal_twocompartment", "vmag", -0.026),
                    ("E7_E8_relax_constitutive_Cvcal_spectrum", "k", 0.033),
                    ("E7_E8_relax_constitutive_Cvcal_spectrum", "M", 0.076),
                    ("E7_E8_relax_constitutive_Cvcal_all", "mu", 0.058),
                    ("E7_E8_relax_constitutive_Cvcal_all", "eta_s", 0.121),
                    ("E7_E8_relax_constitutive_Cvcal_all", "phi", -0.125),
                    ("E7_E8_relax_constitutive_Cvcal_all", "k", -0.052),
                    ("E7_E8_relax_constitutive_Cvcal_all", "M", 0.169)]:
        chk("%s %s" % (c, q), bias[c][q], v)
    chk("viscosity 35 percent low", math.exp(-0.424), 0.654, 0.005, absolute=False)
    push = load(R, "E7_E1_swe_push.json")
    chk("push cost above the noise floor",
        push["map_cost"] / (0.5 * push["n_obs"]), 1.08, 0.02, absolute=False)
    sm = push["summary"]
    chk("push phi factor", sm["phi"]["median"] / sm["phi"]["truth"], 19.0, 0.03,
        absolute=False)
    chk("push k factor", sm["k"]["median"] / sm["k"]["truth"], 1950, 0.03,
        absolute=False)
    for q, v in load(R, "E7_E8_relax_constitutive_Cvcal_spectrum.json")["summary"].items():
        assert v["covered"], q

    # ---- section 9 -------------------------------------------------------
    cov = {r["name"]: r for r in load(R, "coverage.json")}
    for c, q, v in [("E5_joint_network", "mu", 0.910),
                    ("E5_joint_network", "eta_s", 0.945),
                    ("E5_joint_network", "D", 0.900),
                    ("E5_joint_network", "vmag", 0.905),
                    ("E8_relax_constitutive_Cvcal", "mu", 0.850),
                    ("E8_relax_constitutive_Cvcal", "eta_s", 0.933),
                    ("E8_relax_constitutive_Cvcal", "D", 0.908),
                    ("E8_relax_constitutive_Cvcal", "vmag", 0.875),
                    ("E8_relax_independent_matched", "mu", 0.850),
                    ("E7_E1_swe_push", "mu", 0.0),
                    ("E7_E1_swe_push", "eta_s", 0.0),
                    ("E7_E1_swe_push", "phi", 0.0),
                    ("E7_E1_swe_push", "k", 0.0),
                    ("E7_E2_ceus_Acal_twocompartment", "phi", 0.070),
                    ("E7_E2_ceus_Acal_twocompartment", "D", 0.150),
                    ("E7_E2_ceus_Acal_twocompartment", "vmag", 0.000)]:
        chk("coverage %s %s" % (c, q), cov[c]["coverage"][q], v)
    chk("E5 k bias", cov["E5_joint_network"]["mean_bias_log"]["k"], 0.365, 0.01,
        absolute=False)
    chk("E5 k bias as a factor", math.exp(0.365), 1.44, 0.01, absolute=False)
    for c, v in (("E5_joint_network", 200), ("E8_relax_constitutive_Cvcal", 120),
                 ("E7_E1_swe_push", 100),
                 ("E7_E2_ceus_Acal_twocompartment", 100)):
        assert cov[c]["n_rep"] == v, (c, cov[c]["n_rep"])

    # ---- arithmetic quoted in the prose ---------------------------------
    chk("standard error at 120 repetitions", (0.9 * 0.1 / 120) ** 0.5, 0.027,
        0.02, absolute=False)
    chk("coupling gain against the matched control, approximation",
        0.298 / 0.132, 2.3, 0.02, absolute=False)
    chk("normalizing returns k to the matched control",
        con["E8_relax_constitutive_Cvcal_norm"]["k"],
        con["E8_relax_independent_matched"]["k"], 1e-3)
    chk("four percent", 0.352 / 0.339, 1.04, 0.02, absolute=False)
    chk("seven parts in ten thousand", 1 - 0.99927, 7e-4, 0.06, absolute=False)

    print("%d checks, %d failures" % (CHECKS, len(FAILURES)))
    for lab, got, want, d in FAILURES:
        print("   %-46s got %-13.6g quoted %-13.6g deviation %.3g"
              % (lab, got, want, d))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
