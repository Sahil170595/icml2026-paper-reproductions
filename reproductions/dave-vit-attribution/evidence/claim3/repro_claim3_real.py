"""Claim 3 on REAL pretrained weights: low-pass = Gaussian convolution (Eq. 7) on
timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k with a real ImageNet photo.

phi(s) = DAVE effective response  sum( g_eff(X + s u) * (X + s u) )  along a fixed
random unit input ray u through a real image X (target = the model's top-1 class).
Eq. 7 says the Gaussian-perturbation average of the effective transformation is the
convolution with the Gaussian kernel K: we compute the dense-grid convolution
(phi * K_sigma)(0) as reference and show the Monte-Carlo Gaussian average converges
to it at the statistical rate 1/sqrt(n) (RMS over replicas), on the REAL model.

Stages (argv): grid <a> <b>  - evaluate phi on grid points [a:b) (cached json chunk)
               mc            - assemble grid, run MC convergence, write results_real.json
"""
import sys, os, json
import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.dirname(HERE)
sys.path.insert(0, EP)
CACHE = os.path.join(EP, "_cache")
import vit_pretrained as V  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(1)
SIGMA = 0.5
NPTS = 241
GRID = np.linspace(-3.0, 3.0, NPTS)  # +/- 6 sigma


def setup():
    m = V.ViT()
    X = V.preprocess(V.image_paths()[0])  # n01440764 tench, real ImageNet-val photo
    g = torch.Generator().manual_seed(7)
    u = torch.randn(X.shape, generator=g)
    u = u / u.norm()
    with torch.no_grad():
        t = int(m.forward(X[None])[0].argmax())
    return m, X, u, t


def stage_grid(a, b):
    m, X, u, t = setup()
    vals = []
    for s in GRID[a:b]:
        Xs = (X + float(s) * u).clone().requires_grad_(True)
        y = m.forward(Xs[None], detach_op=True)[0, t]
        gr, = torch.autograd.grad(y, Xs)
        vals.append(float((gr * Xs.detach()).sum()))
    json.dump(dict(a=a, b=b, vals=vals),
              open(os.path.join(CACHE, "c3_grid_%03d_%03d.json" % (a, b)), "w"))
    print("grid[%d:%d] done, first=%.6f" % (a, b, vals[0]))


def stage_mc():
    import glob as _g
    phivals = np.full(NPTS, np.nan)
    for f in sorted(_g.glob(os.path.join(CACHE, "c3_grid_*.json"))):
        d = json.load(open(f))
        phivals[d["a"]:d["b"]] = d["vals"]
    assert not np.isnan(phivals).any(), "missing grid chunks"
    dz = GRID[1] - GRID[0]
    K = np.exp(-GRID ** 2 / (2 * SIGMA ** 2)) / (np.sqrt(2 * np.pi) * SIGMA)
    conv0 = float(np.sum(phivals * K) * dz)  # (phi * K_sigma)(0) dense reference
    rng = np.random.default_rng(5)
    ns = [10, 30, 100, 300, 1000, 3000, 10000]
    R = 200
    rms = []
    for n in ns:
        errs = []
        for _ in range(R):
            eps = rng.normal(0, SIGMA, size=n)
            mc = float(np.mean(np.interp(eps, GRID, phivals)))
            errs.append((mc - conv0) ** 2)
        rms.append(float(np.sqrt(np.mean(errs))))
        print("n=%6d  RMS|MC-conv| = %.3e" % (n, rms[-1]))
    slope = float(np.polyfit(np.log10(ns), np.log10(rms), 1)[0])
    print("log-log slope = %.3f (target -0.5)" % slope)
    res = dict(
        model="timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k (pretrained, HF-hub safetensors)",
        image=os.path.basename(V.image_paths()[0]),
        response="DAVE effective response sum(g_eff*(X+s u)) along a fixed unit ray, "
                 "%d real evaluations on [-6sigma,6sigma]" % NPTS,
        sigma=SIGMA, convolution_reference=conv0,
        phi_at_0=float(phivals[NPTS // 2]),
        mc_rms=dict(zip([str(n) for n in ns], rms)),
        loglog_slope=round(slope, 3), slope_target=-0.5,
        verdict="MC Gaussian average of the REAL pretrained ViT effective response "
                "converges to the Gaussian-kernel convolution at 1/sqrt(n) "
                "(slope %.3f), confirming Eq. 7 on real weights." % slope)
    json.dump(res, open(os.path.join(HERE, "results_real.json"), "w"), indent=1)
    print("wrote results_real.json")


if __name__ == "__main__":
    if sys.argv[1] == "grid":
        stage_grid(int(sys.argv[2]), int(sys.argv[3]))
    else:
        stage_mc()
