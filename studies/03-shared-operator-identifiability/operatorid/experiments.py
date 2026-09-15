"""The frozen experiment matrix: cases, arms, controls and coverage.

A *case* fixes the data: which windows, how many contrast frames, what the
sequence corrections generated the data.  An *arm* fixes how the operator is
treated in the inversion.  **Every arm of a case analyzes the same observations
with the same noise scale**, produced once by `generate`; an arm never makes its
own data, because then a difference between arms could come from the data rather
than from the analysis.

The matrix is frozen.  Every case defined here is reported, whether or not it
favors sharing.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np

from .acquisition import Acquisition
from .constants import GAMMA_NOMINAL, psf_widths
from .infer import diagnostics as DG
from .infer import sampling as SA
from .infer import summaries as SU
from .model import GAMMA_NAMES, Inversion, WINDOW_FREE, level_free, WINDOWS
from .params import REPORT, TRUTH, psf_variants, variant

ARMS = ("independent", "shared", "shared_calibrated", "oracle")


@dataclass
class Case:
    name: str
    windows: tuple = WINDOWS
    n_frames_used: int = 0          # 0 means every contrast frame
    #: sequence corrections used to **generate** the data.  Empty means the
    #: nominal ones, so the arms' assumption is correct; the incorrect-sharing
    #: control is exactly the case where this differs from nominal.
    gen_gamma: dict = field(default_factory=dict)
    free_scale: bool = False
    note: str = ""

    def acquisition(self):
        a = Acquisition()
        a.ceus.n_frames_used = self.n_frames_used
        for w in WINDOWS:
            getattr(a, w).enabled = w in self.windows
        return a

    def free(self, arm):
        f = []
        for w in self.windows:
            f += [n for n in WINDOW_FREE[w] if n not in f]
        f += list(level_free(arm))
        if self.free_scale:
            f += ["scale"]
        return tuple(f)

    def overrides(self, arm):
        """Prior overrides.

        The independent arm gets the marginal the shared parameter induces for
        each window, so that the two arms differ in the *relationship* between
        the widths and in nothing else."""
        ov = {}
        if arm == "independent":
            ov.update(psf_variants())
        return ov

    def generating_gamma(self):
        g = dict(GAMMA_NOMINAL)
        g.update(self.gen_gamma)
        return g

    def build(self, arm, truth=None, sigma=None):
        """One arm's inversion.

        Only the oracle is given the generating corrections; every other arm is
        given the nominal ones, which is the assumption under test.  An oracle
        that used the nominal ones in a misspecified case would not be an
        oracle, because it would not know the operator that made the data."""
        gamma = self.generating_gamma() if arm == "oracle" else dict(GAMMA_NOMINAL)
        return Inversion(free=self.free(arm), level=arm, windows=self.windows,
                         acq=self.acquisition(), overrides=self.overrides(arm),
                         truth=dict(truth or TRUTH), gamma=gamma,
                         sigma_fixed=sigma)

    def generator(self, truth=None):
        """The object that makes the data: the true operator, in full."""
        return Inversion(free=(), level="oracle", windows=self.windows,
                         acq=self.acquisition(), truth=dict(truth or TRUTH),
                         gamma=self.generating_gamma())

    def to_dict(self):
        d = asdict(self)
        d["generating_gamma"] = self.generating_gamma()
        return d


def generate(case, seed=0, truth=None):
    """One dataset per case, truth and seed, shared by every arm.

    The noise scale is computed from the **generating** signal.  For the
    contrast window the noise is a fraction of the peak, and the peak depends on
    the point spread, so an arm that computed its own would be analyzing data
    with a different noise level whenever its assumption about the operator was
    wrong.  Two analyses of the same experiment must see the same numbers."""
    gen = case.generator(truth)
    sigma = gen.sigma()
    y = gen.predict(gen.space.x0(gen.truth)) + \
        np.random.default_rng(seed).normal(0.0, sigma)
    return y, sigma


#: The frozen matrix.  Predefined before the main runs and reported in full.
def matrix():
    return [
        Case("main", note="all three windows, every contrast frame, the assumed "
                          "point-spread ratios"),
        Case("one_frame", n_frames_used=1,
             note="information removal: one contrast frame, so the dispersion "
                  "and the point spread are not separable"),
        Case("no_ceus", windows=("bmode", "swe"),
             note="information removal: the contrast window is absent"),
        Case("no_swe", windows=("bmode", "ceus"),
             note="how much the shear window contributes"),
        Case("wrong_ratio", gen_gamma={"ceus": 1.15},
             note="incorrect sharing: the contrast point spread is 15 percent "
                  "wider than the nominal frequency scaling, and the shared arm "
                  "assumes the nominal one"),
        Case("calibrated_scale", free_scale=True,
             note="sensitivity: the reconstructed-to-true length scale is an "
                  "uncertain calibrated measurement, in every arm"),
    ]


# ------------------------------------------------------------------ runner --

def run_arm(case, arm, seed=0, profiles=(), n_draw=20000, data=None):
    """Diagnostics and posterior summary for one arm of one case.

    ``data`` is the (y, sigma) that every arm of this case shares.  It is built
    once by `generate` and passed in; an arm never makes its own."""
    y, sigma = generate(case, seed) if data is None else data
    inv = case.build(arm, sigma=sigma)

    sr = DG.structural_rank(inv)
    sp = DG.spectrum(inv)
    sens = DG.sensitivity_table(inv)
    wsens = DG.window_sensitivity(inv)

    zmap, cost = SA.find_map(inv, y)
    lap = SA.laplace(inv, y, zmap)
    # the reported tissue quantities, plus every operator quantity this arm
    # estimates.  The operator ones matter: the independent arm determines its
    # contrast width very well and its B-mode width not at all, and only both
    # together say what sharing does
    names = [n for n in REPORT if n in inv.free]
    names += [n for n in inv.free if n.startswith(("w_", "w0", "gamma_"))
              and n not in names]
    summ = SU.summarize_gaussian(inv, lap, names)
    # every window's width, for every arm.  Reporting only one would hide the
    # main point of section 3: the independent arm determines its *contrast*
    # width very well and its B-mode width not at all
    summ.update(_width_summary(inv, lap, n_draw))

    out = dict(
        case=case.name, arm=arm, note=case.note, windows=list(case.windows),
        n_frames_used=int(case.n_frames_used),
        generating_gamma=case.generating_gamma(),
        assumed_gamma=dict(GAMMA_NOMINAL), free_scale=bool(case.free_scale),
        free=list(inv.free), n_obs=int(len(y)), seed=int(seed),
        rank=sr["rank"], n_par=sr["n_par"],
        null_labels_log=sr["null_labels_log"],
        singular_ratio=float(sr["singular_values"][-1] / sr["singular_values"][0]),
        eigenvalues=sp["eigenvalues"].tolist(),
        contraction=sp["contraction"].tolist(),
        eig_labels_log=sp["labels_log"],
        sensitivity=sens, window_sensitivity=wsens,
        summary=summ,
        map_z=np.asarray(zmap).tolist(), map_cost=float(cost),
        posterior_correlation=_corr(lap["cov"]).tolist(),
    )
    if profiles:
        out["profiles"] = {n: _profile(inv, y, n) for n in profiles
                           if n in inv.free}
    return out


#: Physical ranges the profiles are taken over, in the natural units of each
#: quantity.
#:
#: A profile over plus and minus a few *prior* standard deviations is useless
#: here.  The prior on ``sig_e`` spans a factor of forty, while the B-mode data
#: pin ``sqrt(sig_e^2 + w^2)`` tightly, so the point spread can only absorb a
#: change in the margin width until ``sig_e`` reaches that quadrature sum, about
#: 0.70 mm at the truth.  Beyond it no compensation exists and the profile
#: saturates at a value that says nothing about the arms.  The range below stays
#: inside the region where the compensation is possible, which is exactly the
#: region where the arms can differ.
PROFILE_RANGE = {"sig_e": (0.30e-3, 0.68e-3)}


def _profile(inv, y, name, n=11):
    lo, hi = PROFILE_RANGE.get(name, (None, None))
    if lo is None:
        pr0 = dict(grid=None)
        grid = None
    else:
        e = inv.space.entry(name)
        m, sd = DG.standardize(inv)
        i = inv.space.index(name)
        grid = (np.linspace(e.to_x(lo), e.to_x(hi), n) - m[i]) / sd[i]
    rec = {}
    for level in DG.PROFILE_LEVELS:
        pr = DG.profile_likelihood(inv, y, name, grid=grid, n=n, span=3.0,
                                   level=level)
        x = DG.z_to_x(inv, np.asarray(pr["z_hat"]))
        rec["grid"] = np.asarray(pr["grid"]).tolist()
        rec["value"] = [float(inv.space.entry(name).to_value(v))
                        for v in x[:, inv.space.index(name)]]
        rec["delta_" + level] = np.asarray(pr["delta"]).tolist()
        rec["audit_" + level] = DG.profile_audit(inv, pr)
        # what the fit did with the point spread at each grid point.  This is
        # the whole mechanism in two columns: in the independent arm the B-mode
        # width slides to absorb the forced change, in the shared arm it cannot,
        # because moving it would move the contrast width too.
        for w in WINDOWS:
            rec["width_%s_%s" % (w, level)] = [
                float(inv.width(inv.physical(xi), w)) for xi in x]
        if name == "sig_e":
            rec["quadrature_%s" % level] = [
                float(np.hypot(v, wb)) for v, wb in
                zip(rec["value"], rec["width_bmode_%s" % level])]
    rec["range"] = [lo, hi]
    return rec


#: operator quantities summarized for every arm, however that arm parameterizes
#: the operator
OPERATOR_REPORT = ("w0", "w_bmode", "w_swe", "w_ceus")


def _width_summary(inv, lap, n_draw=20000, seed=7):
    """Posterior summary of every point-spread width, however the arm
    parameterizes the operator, so the arms can be compared on one axis.

    All four are reported because the interesting statement is not about one of
    them: the independent arm pins its contrast width and not its B-mode width,
    and only the two together say what sharing does."""
    rng = np.random.default_rng(seed)
    truth_w = dict(psf_widths(inv.truth["w0"], inv.gamma))
    truth_w["w0"] = inv.truth["w0"]
    if len(inv.free) == 0 or inv.level == "oracle":
        return {n: dict(median=float(truth_w[n.replace("w_", "") if n != "w0" else "w0"]),
                        lo=float(truth_w[n.replace("w_", "") if n != "w0" else "w0"]),
                        hi=float(truth_w[n.replace("w_", "") if n != "w0" else "w0"]),
                        width_post=0.0, width_prior=float("nan"),
                        contraction=0.0,
                        truth=float(truth_w[n.replace("w_", "") if n != "w0" else "w0"]),
                        bias_log=0.0, covered=True,
                        note="held at the generating value")
                for n in OPERATOR_REPORT}
    zs = rng.multivariate_normal(lap["mean"], lap["cov"], n_draw)
    zp = SA.sample_prior(inv, rng, n_draw)
    out = {}
    for n in OPERATOR_REPORT:
        if n in inv.free:
            continue                       # already in the ordinary summary
        post = np.array([inv.quantity(DG.z_to_x(inv, z), n) for z in zs])
        pri = np.array([inv.quantity(DG.z_to_x(inv, z), n) for z in zp])
        lo, med, hi = np.percentile(post, [5, 50, 95])
        plo, phi = np.percentile(pri, [5, 95])
        t = truth_w["w0" if n == "w0" else n.replace("w_", "")]
        wp, wq = np.log(hi) - np.log(lo), np.log(phi) - np.log(plo)
        out[n] = dict(median=float(med), lo=float(lo), hi=float(hi),
                      prior_lo=float(plo), prior_hi=float(phi),
                      width_post=float(wp), width_prior=float(wq),
                      contraction=float(wp / wq) if wq > 0 else float("nan"),
                      truth=float(t), bias_log=float(np.log(med / t)),
                      covered=bool(lo <= t <= hi))
    return out


def _corr(c):
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


# ----------------------------------------------------------------- coverage -

#: Three predefined truths, differing in the quantity the mechanism targets.
#: Modest by design: this is a calibration check, not a validation campaign.
COVERAGE_TRUTHS = (
    ("sharp", {"sig_e": 0.30e-3}),
    ("nominal", {}),
    ("diffuse", {"sig_e": 1.20e-3}),
)


def coverage(case, arm, truth_name, truth_over, n_rep=40, seed=0):
    """Empirical coverage and bias over repeated noise, at one truth.

    Smaller intervals are only better if they still contain the truth, so the
    bias and the coverage are reported next to the width and never instead of
    it."""
    truth = dict(TRUTH, **truth_over)
    _, sigma = generate(case, seed, truth)
    inv = case.build(arm, truth=truth, sigma=sigma)
    names = [n for n in REPORT if n in inv.free]
    hits = {n: 0 for n in names}
    bias = {n: [] for n in names}
    width = {n: [] for n in names}
    gen = case.generator(truth)
    x_gen = gen.space.x0(gen.truth)
    y0 = gen.predict(x_gen)
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n_rep):
        y = y0 + rng.normal(0.0, sigma)
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
            width[n].append(s[n]["width_post"])
    f = lambda d, g: {n: float(g(d[n])) if d[n] else float("nan") for n in names}
    return dict(case=case.name, arm=arm, truth=truth_name,
                truth_overrides=truth_over, n_rep=int(ok),
                coverage={n: hits[n] / max(ok, 1) for n in names},
                mean_bias_log=f(bias, np.mean),
                sd_bias_log=f(bias, np.std),
                mean_width_log=f(width, np.mean))
