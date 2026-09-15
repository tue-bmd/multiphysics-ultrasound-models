"""The experiment matrix.

Every configuration uses the same synthetic truth, the same marginal priors and,
where configurations are compared, the same noise realization, so that
differences between them come from the configuration and not from the draw.

Identifiers follow METHODS.md.  Suffixes:
  ``_norm``   normalized CEUS curves instead of absolute occupancy
  ``_Acal``   delivered tracer amount independently calibrated
  ``_Pnui``   applied compressive stress not measured
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np

from .acquisition import Acquisition
from .coupling import CorrelatedPrior, extra_free
from .model import Inversion
from .params import variant
from .models.richer import Richer
from .infer import diagnostics as DG
from .infer import sampling as SA
from .infer import summaries as SU

#: free parameters contributed by each window
WINDOW_FREE = {
    "swe": ("mu", "eta_s", "F0", "Tp"),
    "ceus": ("D", "vmag", "vth", "vaz", "A", "t0"),
    # L enters the relaxation time as L^2, so holding it at its truth while
    # calling it calibrated would hide a real uncertainty
    "relax": ("M", "P", "L"),
}
#: phi is shared: it is a tissue parameter that both the shear window and the
#: contrast window could in principle see
SHARED = ("phi", "k")

REPORT = ("mu", "eta_s", "phi", "k", "D", "vmag", "S_v", "C_v", "M", "A", "t0")

#: A calibrated to 10 percent: an independently measured delivered amount
A_CAL = dict(A=variant("A", lo=1 / 1.1, hi=1.1, role="calibrated"))
#: applied stress not measured, only bounded
P_NUI = dict(P=variant("P", lo=100.0, hi=2500.0, role="nuisance"))
#: fractional vascular compliance measured independently, to a factor of 1.5
CV_CAL = dict(C_v=variant("C_v", lo=1.0e-3 / 1.5, hi=1.0e-3 * 1.5,
                          role="calibrated"))


def matched_Sv(cv_entry=None):
    """An independent ``S_v`` prior with the marginal that ``phi * C_v`` induces.

    Comparing the coupled model against an independent one is only a comparison
    of *coupling* if the two carry the same marginal information about the
    storage.  When the compliance is calibrated, the coupled model's induced
    marginal on S_v is much narrower than the broad-compliance one, so the
    default independent arm is not a matched control for it: the difference
    would include the calibration as well as the coupling.  This builds the
    independent arm that matches whichever compliance prior is in force.

    **This is matched prior information, not a storage measurement.**  Its role
    is ``matched``, which is not ``calibrated``: nothing here is measured
    independently of the imaging observations, and no auxiliary experiment
    determines the storage.  The marginal is constructed to equal the one the
    coupled arm induces, and that is all.  The two roles enter the same profile
    level because a comparison requires the same information to be counted at
    the same level in both arms (see `infer.diagnostics.AUXILIARY_ROLES`), but
    naming them alike would claim an experiment that does not exist."""
    from .params import CATALOGUE, Z90
    a = CATALOGUE["phi"]
    b = cv_entry or CATALOGUE["C_v"]
    m = a.mean + b.mean
    sd = float(np.hypot(a.sd, b.sd))
    return dict(S_v=variant("S_v",
                            lo=float(np.exp(m - Z90 * sd)),
                            hi=float(np.exp(m + Z90 * sd)),
                            nominal=a.nominal * b.nominal,
                            role="matched"))


#: the independent arm matched to the calibrated-compliance coupled arm
SV_MATCHED_CVCAL = matched_Sv(CV_CAL["C_v"])


@dataclass
class Config:
    name: str
    windows: tuple
    level: str = "independent"
    normalized: bool = False
    overrides: dict = field(default_factory=dict)
    richer: bool = False
    richer_kwargs: dict = field(default_factory=dict)
    note: str = ""

    def free(self):
        f = list(SHARED)
        for w in self.windows:
            f += [n for n in WINDOW_FREE[w] if n not in f]
        f += extra_free(self.level)
        return tuple(f)

    def build(self, corr=None, seed=0):
        acq = Acquisition()
        acq.ceus.normalized = self.normalized
        acq.relax.enabled = "relax" in self.windows
        inv = Inversion(free=self.free(), level=self.level, windows=self.windows,
                        acq=acq, overrides=dict(self.overrides),
                        corr=corr if self.level == "network" else None)
        return inv


def matrix(include_relax=True):
    """The configurations, in reporting order."""
    c = []
    # --- E1, E2: single modality -----------------------------------------
    c.append(Config("E1_swe", ("swe",), note="shear window alone"))
    c.append(Config("E2_ceus", ("ceus",), note="absolute occupancy alone"))
    c.append(Config("E2n_ceus_norm", ("ceus",), normalized=True,
                    note="normalized curves alone"))
    c.append(Config("E2_ceus_Acal", ("ceus",), overrides=dict(A_CAL),
                    note="absolute occupancy, delivered amount calibrated"))
    # --- E3 to E5: current protocol, three coupling levels ----------------
    for i, lvl in enumerate(("independent", "constitutive", "network")):
        c.append(Config("E%d_joint_%s" % (3 + i, lvl),
                        ("swe", "ceus"), level=lvl, overrides=dict(A_CAL),
                        note="current protocol, %s coupling" % lvl))
    ov = dict(A_CAL); ov.update(CV_CAL)
    c.append(Config("E4_joint_constitutive_Cvcal", ("swe", "ceus"),
                    level="constitutive", overrides=dict(ov),
                    note="current protocol, constitutive coupling, compliance calibrated"))
    # the network route, with the contrast window alone: if this matches the
    # joint configuration then the shear window contributes nothing to it
    c.append(Config("E5_ceus_only_network", ("ceus",), level="network",
                    overrides=dict(A_CAL),
                    note="contrast window alone, network-informed prior"))
    # --- E6: same, normalized --------------------------------------------
    for lvl in ("independent", "constitutive", "network"):
        c.append(Config("E6_joint_%s_norm" % lvl, ("swe", "ceus"), level=lvl,
                        normalized=True, overrides=dict(A_CAL),
                        note="current protocol, %s coupling, normalized CEUS" % lvl))
    # --- E8: contingency, relaxation added --------------------------------
    if include_relax:
        ov = dict(A_CAL); ov.update(P_NUI)
        for lvl in ("independent", "constitutive", "network"):
            c.append(Config("E8_relax_%s" % lvl, ("swe", "ceus", "relax"),
                            level=lvl, overrides=dict(ov),
                            note="contingency, %s coupling, stress not measured" % lvl))
        c.append(Config("E8_relax_constitutive_norm", ("swe", "ceus", "relax"),
                        level="constitutive", normalized=True, overrides=dict(ov),
                        note="contingency, constitutive coupling, normalized CEUS"))
        c.append(Config("E8_relax_constitutive_Pcal", ("swe", "ceus", "relax"),
                        level="constitutive", overrides=dict(A_CAL),
                        note="contingency, constitutive coupling, stress calibrated"))
        ov2 = dict(ov); ov2.update(CV_CAL)
        for lvl in ("constitutive", "network"):
            c.append(Config("E8_relax_%s_Cvcal" % lvl, ("swe", "ceus", "relax"),
                            level=lvl, overrides=dict(ov2),
                            note="contingency, %s coupling, compliance calibrated" % lvl))
        # the matched control for the line above: independent, but carrying the
        # same marginal information about the storage that phi times a
        # calibrated C_v induces.  Without it the comparison credits the
        # calibration to the coupling.
        ov3 = dict(ov); ov3.update(SV_MATCHED_CVCAL)
        c.append(Config("E8_relax_independent_matched", ("swe", "ceus", "relax"),
                        level="independent", overrides=dict(ov3),
                        note=("contingency, independent priors, storage marginal "
                              "matched to the calibrated-compliance coupled arm")))
        ov3 = dict(ov2); ov3["C_v"] = CV_CAL["C_v"]
        c.append(Config("E8_relax_constitutive_Cvcal_norm", ("swe", "ceus", "relax"),
                        level="constitutive", normalized=True, overrides=dict(ov3),
                        note=("contingency, constitutive coupling, compliance "
                              "calibrated, normalized CEUS")))
    return c


# ------------------------------------------------------------------ runner --

def run_config(cfg, corr=None, seed=0, mcmc=False, profiles=(), n_draw=20000,
               truth=None, report=None):
    """Diagnostics and posterior summary for one configuration."""
    inv = cfg.build(corr, seed)
    if truth is not None:
        inv.truth.update({k: float(v) for k, v in truth.items()})
    gen = Richer(**cfg.richer_kwargs) if cfg.richer else None
    rng = np.random.default_rng(seed)
    y = inv.simulate(rng, gen=gen)

    sr = DG.structural_rank(inv)
    sp = DG.spectrum(inv)
    sens = DG.sensitivity_table(inv)
    wsens = DG.window_sensitivity(inv)

    zmap, cost = SA.find_map(inv, y)
    lap = SA.laplace(inv, y, zmap)
    # report a quantity only when it is free or is determined by free ones;
    # a quantity held at its truth has zero prior width and no contraction
    names = [n for n in (REPORT if report is None else tuple(report))
             if n in inv.free or (n == "S_v" and cfg.level != "independent")]
    summ = SU.summarize_gaussian(inv, lap, names)

    out = dict(
        name=cfg.name, note=cfg.note, windows=list(cfg.windows), level=cfg.level,
        normalized=bool(cfg.normalized), richer=bool(cfg.richer),
        free=list(inv.free), n_obs=int(len(y)), seed=int(seed),
        overrides={k: dict(lo=v.lo, hi=v.hi, role=v.role)
                   for k, v in cfg.overrides.items()},
        rank=sr["rank"], n_par=sr["n_par"],
        singular_ratio=float(sr["singular_values"][-1] / sr["singular_values"][0]),
        null_labels_log=sr["null_labels_log"],
        eigenvalues=sp["eigenvalues"].tolist(),
        contraction=sp["contraction"].tolist(),
        eig_labels_z=sp["labels"], eig_labels_log=sp["labels_log"],
        sensitivity=sens, window_sensitivity=wsens,
        summary=summ,
        map_z=np.asarray(zmap).tolist(), map_cost=float(cost),
        posterior_correlation=_corr_from_cov(lap["cov"]).tolist(),
    )
    if profiles:
        out["profiles"] = {n: {k: np.asarray(v).tolist() for k, v in
                               DG.profile_likelihood(inv, y, n).items()}
                           for n in profiles if n in inv.free}
    if mcmc:
        mc = SA.adaptive_metropolis(inv, y, n_steps=mcmc, seed=seed)
        zp = SA.sample_prior(inv, np.random.default_rng(seed + 2), n_draw)
        out["mcmc"] = dict(rhat=mc["rhat"].tolist(), ess=mc["ess"].tolist(),
                           accept=mc["accept"],
                           summary=SA.summarize(inv, mc["samples"], zp, names))
    return out


def _corr_from_cov(c):
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def coverage(cfg, corr=None, n_rep=100, seed=0, n_draw=4000):
    """Empirical coverage of the 90 percent credible interval over repeated
    noise realizations, with the Gaussian approximation of the posterior."""
    inv = cfg.build(corr, seed)
    gen = Richer(**cfg.richer_kwargs) if cfg.richer else None
    names = [n for n in REPORT
             if n in inv.free or (n == "S_v" and cfg.level != "independent")]
    hits = {n: 0 for n in names}
    bias = {n: [] for n in names}
    rng = np.random.default_rng(seed)
    ok = 0
    for i in range(n_rep):
        y = inv.simulate(rng, gen=gen)
        try:
            zmap, _ = SA.find_map(inv, y, restarts=1)
            lap = SA.laplace(inv, y, zmap)
            s = SU.summarize_gaussian(inv, lap, names)
        except Exception:
            continue
        ok += 1
        for n in names:
            if n not in s:
                continue
            hits[n] += int(s[n]["covered"])
            bias[n].append(s[n]["bias_log"])
    return dict(name=cfg.name, n_rep=int(ok),
                coverage={n: hits[n] / max(ok, 1) for n in names},
                mean_bias_log={n: float(np.mean(bias[n])) if bias[n] else float("nan")
                               for n in names},
                sd_bias_log={n: float(np.std(bias[n])) if bias[n] else float("nan")
                             for n in names})


#: the model-discrepancy experiment, one departure at a time and then combined
DEPARTURES = {
    "push": dict(slow_fraction=0.0, k_spread=0.0),
    "twocompartment": dict(push_width=0.0, k_spread=0.0),
    "spectrum": dict(push_width=0.0, slow_fraction=0.0),
    "all": dict(),
}


def discrepancy_matrix():
    """Richer generating models fitted by the reduced inversion model.

    Each departure is introduced on its own before all of them together, so that
    a bias can be attributed to a particular departure rather than to an
    unspecified mismatch."""
    base = {c.name: c for c in matrix()}
    out = []
    for host, dep in (("E1_swe", "push"),
                      ("E2_ceus_Acal", "twocompartment"),
                      ("E8_relax_constitutive_Cvcal", "spectrum"),
                      ("E8_relax_constitutive_Cvcal", "all")):
        c = base[host]
        out.append(Config("E7_%s_%s" % (host, dep), c.windows, c.level,
                          c.normalized, dict(c.overrides), richer=True,
                          richer_kwargs=dict(DEPARTURES[dep]),
                          note="%s, generated with the %s departure" % (c.note, dep)))
    return out
