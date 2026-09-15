"""Check every number quoted in RESULTS.md against the files it cites.

    python scripts/verify_results.py [--results results]

This exists so that a reader does not have to take the prose on trust, and so
that a change to the code that moves a reported number fails loudly rather than
silently leaving the document wrong.  It reads only the machine-readable
outputs; it does not re-run anything.
"""
from __future__ import annotations
import argparse
import json
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


ARMS = ("independent", "shared", "shared_calibrated", "oracle")


def arm(root, case, a):
    return load(root, "%s__%s.json" % (case, a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = a.results

    # ---- section 1: the setting -----------------------------------------
    sc = load(R, "screening.json")
    for w0, want in (("0.15 mm", dict(bmode=1.000, swe=3.012, ceus=2.500)),
                     ("0.35 mm", dict(bmode=1.000, swe=1.807, ceus=2.500)),
                     ("0.80 mm", dict(bmode=1.000, swe=1.487, ceus=2.500))):
        for m, v in want.items():
            chk("width ratio at %s, %s" % (w0, m),
                sc["width_map"][w0][m]["ratio"], v, 0.002, absolute=False)
    # the shear ratio is not a constant, which is the point
    r = [sc["width_map"][k]["swe"]["ratio"] for k in ("0.15 mm", "0.80 mm")]
    assert r[0] / r[1] > 2.0, r
    for k, v in {
            "blur over margin width, w_B / sig_e": 0.583,
            "quadrature sum over margin, sig_obs / sig_e": 1.158,
            "profile samples per observed margin": 6.9,
            "shear wavelength (mm)": 10.0,
            "aperture in wavelengths": 2.0,
            "blur attenuation at f0, exp(-(k w)^2/2)": 0.925,
            "medium attenuation over the aperture": 0.449,
            "blur share of variance, first frame": 0.657,
            "blur share of variance, last frame": 0.154}.items():
        chk("group " + k, sc["groups"][k], v, 0.01, absolute=False)
    for k, v in dict(scale=172.0, gamma_ceus=10.9, Y_atten=1.14,
                     gamma_swe=0.91, track_kernel=0.61).items():
        chk("fixed screening " + k, sc["fixed_screening"][k], v, 0.02,
            absolute=False)
    jc = sc["jacobian_step_convergence"]
    for got, want in zip(jc[:3], (1.9e-4, 4.4e-5, 8.8e-6)):
        chk("jacobian convergence", got, want, 0.1, absolute=False)
    assert jc[0] > jc[1] > jc[2], jc

    # ---- section 2: the confounding -------------------------------------
    ind = arm(R, "main", "independent")
    assert "sig_e^-0.32 w_bmode^+0.95" in ind["null_labels_log"], \
        ind["null_labels_log"]
    chk("B-mode degeneracy is exact", ind["singular_ratio"], 0.0, 1e-8)
    ws = arm(R, "main", "shared")["window_sensitivity"]
    for w, v in (("ceus", 71.3), ("swe", 5.6), ("bmode", 5.7)):
        chk("w0 sensitivity, %s" % w, ws[w]["w0"], v, 0.02, absolute=False)
    assert ws["ceus"]["w0"] > 5 * ws["swe"]["w0"]

    # ---- sections 3 and 4: the comparison and the four widths -----------
    expect = {
        "independent":       dict(sig_e=0.190, w_bmode=0.982, w_swe=0.272, w_ceus=0.023),
        "shared":            dict(sig_e=0.038, w_bmode=0.023, w_swe=0.024, w_ceus=0.023),
        "shared_calibrated": dict(sig_e=0.042, w_bmode=0.088, w_swe=0.128, w_ceus=0.023),
    }
    for a_, qs in expect.items():
        s = arm(R, "main", a_)["summary"]
        for q, v in qs.items():
            chk("main %s %s" % (a_, q), s[q]["contraction"], v, 0.03,
                absolute=False)
    chk("main oracle sig_e",
        arm(R, "main", "oracle")["summary"]["sig_e"]["contraction"], 0.037,
        0.03, absolute=False)
    # the statement of section 4: the contrast width is equally well determined
    # in both arms, and only the B-mode width differs
    ci = arm(R, "main", "independent")["summary"]["w_ceus"]["contraction"]
    cs = arm(R, "main", "shared")["summary"]["w_ceus"]["contraction"]
    chk("the contrast width is equally determined in both arms", ci, cs, 0.05,
        absolute=False)
    bi = arm(R, "main", "independent")["summary"]["w_bmode"]["contraction"]
    bs = arm(R, "main", "shared")["summary"]["w_bmode"]["contraction"]
    assert bi / bs > 30, (bi, bs)

    P = {a_: arm(R, "main", a_)["profiles"]["sig_e"] for a_ in ARMS}
    prof = {a_: (max(P[a_]["delta_imaging"]),
                 max(P[a_]["delta_imaging_plus_calibration"])) for a_ in ARMS}
    chk("independent likelihood profile is exactly flat", prof["independent"][0],
        0.0, 1e-6)
    chk("shared likelihood profile", prof["shared"][0], 121.9, 0.02,
        absolute=False)
    chk("oracle likelihood profile", prof["oracle"][0], 135.4, 0.02,
        absolute=False)
    # the calibrated arm: flat at the likelihood level, restored by calibration
    chk("calibrated arm is flat at the likelihood level",
        prof["shared_calibrated"][0], 0.0, 1e-6)
    chk("calibrated arm with its calibration", prof["shared_calibrated"][1],
        61.1, 0.02, absolute=False)
    for a_ in ARMS:
        au = P[a_]["audit_imaging"]
        assert au["n_not_converged"] == 0, (a_, au["n_not_converged"])
        assert au["parameters_on_a_bound"] == [], (a_, au)
        for q, v in au["per_parameter"].items():
            assert v["outside_domain"] == 0, (a_, q)
    chk("profile range low", P["shared"]["range"][0], 0.30e-3, 1e-9)
    chk("profile range high", P["shared"]["range"][1], 0.68e-3, 1e-9)

    want_rows = {
        "independent": [(0, 0.30, 0.622, 0.691, 0.894, 0.00),
                        (5, 0.45, 0.523, 0.691, 0.894, 0.00),
                        (10, 0.68, 0.121, 0.691, 0.894, 0.00)],
        "shared": [(0, 0.30, 0.371, 0.477, 0.926, 121.9),
                   (5, 0.45, 0.363, 0.579, 0.906, 29.6),
                   (10, 0.68, 0.356, 0.767, 0.890, 11.7)],
        "shared_calibrated": [(0, 0.30, 0.622, 0.691, 0.894, 0.00),
                              (5, 0.45, 0.523, 0.691, 0.894, 0.00),
                              (10, 0.68, 0.121, 0.691, 0.894, 0.00)],
        "oracle": [(0, 0.30, 0.350, 0.461, 0.875, 135.4)],
    }
    for a_, rows in want_rows.items():
        p = P[a_]
        for i, se, wb, q, wc, pen in rows:
            chk("%s row %d sig_e" % (a_, i), p["value"][i] * 1e3, se, 0.01)
            chk("%s row %d w_B" % (a_, i),
                p["width_bmode_imaging"][i] * 1e3, wb, 0.002)
            chk("%s row %d quadrature" % (a_, i),
                p["quadrature_imaging"][i] * 1e3, q, 0.002)
            chk("%s row %d w_C" % (a_, i),
                p["width_ceus_imaging"][i] * 1e3, wc, 0.002)
            chk("%s row %d likelihood" % (a_, i), p["delta_imaging"][i], pen,
                0.06)
    for a_ in ("independent", "shared_calibrated"):
        qi = P[a_]["quadrature_imaging"]
        chk("%s holds the quadrature sum" % a_, max(qi) / min(qi) - 1.0, 0.0,
            1e-5)
    wb = P["shared"]["width_bmode_imaging"]
    chk("the shared B-mode blur stays within five percent",
        max(wb) / min(wb) - 1.0, 0.0, 0.05)
    assert max(P["shared"]["quadrature_imaging"]) \
        / min(P["shared"]["quadrature_imaging"]) > 1.5
    wo = P["oracle"]["width_bmode_imaging"]
    chk("the oracle blur is held", max(wo) - min(wo), 0.0, 1e-15)
    # the calibrated arm's own profile for the calibration level
    chk("calibrated arm, forced 0.30 mm, with calibration",
        P["shared_calibrated"]["delta_imaging_plus_calibration"][0], 61.1, 0.02,
        absolute=False)

    # ---- section 5: what supplies the information -----------------------
    for case, v in (("main", 0.038), ("one_frame", 0.075), ("no_ceus", 0.075),
                    ("no_swe", 0.037)):
        chk("%s shared sig_e" % case,
            arm(R, case, "shared")["summary"]["sig_e"]["contraction"], v, 0.02,
            absolute=False)
    for case, v in (("main", 0.023), ("one_frame", 0.252), ("no_ceus", 0.250),
                    ("no_swe", 0.024)):
        chk("%s shared w_bmode" % case,
            arm(R, case, "shared")["summary"]["w_bmode"]["contraction"], v,
            0.03, absolute=False)
    chk("one frame equals no contrast window",
        arm(R, "one_frame", "shared")["summary"]["sig_e"]["contraction"],
        arm(R, "no_ceus", "shared")["summary"]["sig_e"]["contraction"], 2e-3)
    for case, v in (("one_frame", 16.1), ("no_ceus", 16.1), ("no_swe", 129.7)):
        chk("%s shared profile" % case,
            max(arm(R, case, "shared")["profiles"]["sig_e"]["delta_imaging"]),
            v, 0.02, absolute=False)
    # with one frame the contrast window cannot determine its own width either
    chk("one frame: the contrast width is not determined",
        arm(R, "one_frame", "independent")["summary"]["w_ceus"]["contraction"],
        0.986, 0.02, absolute=False)

    # ---- section 6: misspecification ------------------------------------
    for a_, v in (("shared", 0.043), ("shared_calibrated", 0.050)):
        chk("wrong ratio %s sig_e" % a_,
            arm(R, "wrong_ratio", a_)["summary"]["sig_e"]["contraction"], v,
            0.03, absolute=False)
    chk("wrong ratio generating correction",
        arm(R, "wrong_ratio", "shared")["generating_gamma"]["ceus"], 1.15, 1e-12)
    chk("the other arms assume the nominal one",
        arm(R, "wrong_ratio", "shared")["assumed_gamma"]["ceus"], 1.0, 1e-12)

    # ---- section 7: coverage --------------------------------------------
    cov = {(r["case"], r["arm"], r["truth"]): r for r in load(R, "coverage.json")}
    want = [
        ("main", "independent", "sharp", 1.000, 0.153, 1.448),
        ("main", "shared", "sharp", 0.850, -0.006, 0.298),
        ("main", "shared_calibrated", "sharp", 0.975, 0.001, 0.384),
        ("main", "oracle", "sharp", 0.900, -0.005, 0.289),
        ("main", "independent", "nominal", 1.000, -0.001, 0.701),
        ("main", "shared", "nominal", 0.875, -0.007, 0.138),
        ("main", "shared_calibrated", "nominal", 0.900, -0.006, 0.151),
        ("main", "oracle", "nominal", 0.900, -0.007, 0.136),
        ("main", "independent", "diffuse", 1.000, -0.006, 0.206),
        ("main", "shared", "diffuse", 0.825, -0.005, 0.090),
        ("main", "shared_calibrated", "diffuse", 0.850, -0.005, 0.092),
        ("main", "oracle", "diffuse", 0.850, -0.005, 0.090),
        ("wrong_ratio", "shared", "nominal", 0.700, -0.065, 0.155),
        ("wrong_ratio", "shared_calibrated", "nominal", 0.775, -0.058, 0.176),
        ("wrong_ratio", "shared", "sharp", 0.450, -0.293, 0.551),
        ("wrong_ratio", "shared_calibrated", "sharp", 1.000, -0.226, 0.678),
        ("wrong_ratio", "independent", "nominal", 1.000, -0.001, 0.701),
        ("wrong_ratio", "oracle", "nominal", 0.900, -0.007, 0.136),
    ]
    for c, a_, t, cv, bi_, wd in want:
        r = cov[(c, a_, t)]
        assert r["n_rep"] == 40, (c, a_, t, r["n_rep"])
        chk("coverage %s %s %s" % (c, a_, t), r["coverage"]["sig_e"], cv, 0.001)
        chk("bias %s %s %s" % (c, a_, t), r["mean_bias_log"]["sig_e"], bi_,
            0.002)
        chk("width %s %s %s" % (c, a_, t), r["mean_width_log"]["sig_e"], wd,
            0.01, absolute=False)
    chk("standard error at 40 repetitions", (0.9 * 0.1 / 40) ** 0.5, 0.047,
        0.02, absolute=False)
    chk("the shared arm's bias grows by a factor of 9.8",
        cov[("wrong_ratio", "shared", "nominal")]["mean_bias_log"]["sig_e"]
        / cov[("main", "shared", "nominal")]["mean_bias_log"]["sig_e"],
        9.8, 0.03, absolute=False)
    # the independent arm is untouched by the misspecification
    chk("the independent arm is untouched",
        cov[("wrong_ratio", "independent", "nominal")]["mean_width_log"]["sig_e"],
        cov[("main", "independent", "nominal")]["mean_width_log"]["sig_e"],
        1e-5, absolute=False)
    # and estimating the corrections buys coverage back
    assert cov[("wrong_ratio", "shared_calibrated", "sharp")]["coverage"]["sig_e"] \
        > cov[("wrong_ratio", "shared", "sharp")]["coverage"]["sig_e"] + 0.4

    # ---- section 8: sampling validation ---------------------------------
    mc = {r["arm"]: r for r in load(R, "mcmc_validation.json")["configurations"]}
    for a_, steps, rh, eb, et in (("independent", 165000, 1.0044, 687, 505),
                                  ("shared", 60000, 1.0016, 2902, 4901)):
        r = mc[a_]
        assert r["sampler"]["n_steps"] == steps, (a_, r["sampler"]["n_steps"])
        chk("rank R-hat %s" % a_, r["worst_rhat_rank"], rh, 0.002)
        chk("bulk ESS %s" % a_, r["min_ess_bulk"], eb, 0.05, absolute=False)
        chk("tail ESS %s" % a_, r["min_ess_tail"], et, 0.05, absolute=False)
        assert r["worst_rhat_rank"] < 1.01, a_
    for a_, lo, hi, ratio in (("independent", 0.149e-3, 0.685e-3, 0.46),
                              ("shared", 0.549e-3, 0.632e-3, 0.99)):
        c = mc[a_]["comparison"]["sig_e"]
        chk("sampled lo %s" % a_, c["sampled"]["lo"], lo, 0.01, absolute=False)
        chk("sampled hi %s" % a_, c["sampled"]["hi"], hi, 0.01, absolute=False)
        chk("width ratio %s" % a_, c["width_ratio_gaussian_over_sampled"],
            ratio, 0.02, absolute=False)
    for a_ in ("independent", "shared"):
        for q, c in mc[a_]["comparison"].items():
            if q == "sig_e":
                continue
            assert abs(c["width_ratio_gaussian_over_sampled"] - 1) < 0.04, (a_, q)
    # the per-chain spread is the reason the headline is not quoted to two
    # significant figures
    pc = mc["independent"]["per_chain_width"]["sig_e"]
    chk("independent per-chain spread", pc["spread_factor"], 1.33, 0.03,
        absolute=False)
    assert mc["shared"]["per_chain_width"]["sig_e"]["spread_factor"] < 1.05
    ef = load(R, "mcmc_validation.json")["effect"]
    chk("sampled independent", ef["sampled"]["independent"], 0.413, 0.02,
        absolute=False)
    chk("sampled shared", ef["sampled"]["shared"], 0.038, 0.02, absolute=False)
    chk("sampled ratio", ef["sampled"]["ratio"], 10.9, 0.02, absolute=False)
    chk("gaussian ratio", ef["gaussian"]["ratio"], 5.07, 0.02, absolute=False)
    assert ef["sampled"]["ratio"] > ef["gaussian"]["ratio"]
    # log space and millimetres are different numbers, and the document says so
    c = mc["independent"]["comparison"]["sig_e"]
    d = mc["shared"]["comparison"]["sig_e"]
    mm = ((c["sampled"]["hi"] - c["sampled"]["lo"])
          / (d["sampled"]["hi"] - d["sampled"]["lo"]))
    chk("the ratio in millimetres", mm, 6.4, 0.03, absolute=False)

    # ---- section 9: the calibration sensitivity -------------------------
    cs_i = arm(R, "calibrated_scale", "independent")["summary"]["sig_e"]
    cs_s = arm(R, "calibrated_scale", "shared")["summary"]["sig_e"]
    chk("calibrated scale independent", cs_i["contraction"], 0.190, 0.02,
        absolute=False)
    chk("calibrated scale shared", cs_s["contraction"], 0.039, 0.03,
        absolute=False)
    csp = arm(R, "calibrated_scale", "shared")["profiles"]["sig_e"]
    chk("with the scale free the likelihood is flat in every arm",
        max(csp["delta_imaging"]), 0.0, 1e-6)
    assert max(csp["delta_imaging_plus_calibration"]) > 100

    print("%d checks, %d failures" % (CHECKS, len(FAILURES)))
    for lab, got, want_, d_ in FAILURES:
        print("   %-46s got %-13.6g quoted %-13.6g deviation %.3g"
              % (lab, got, want_, d_))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
