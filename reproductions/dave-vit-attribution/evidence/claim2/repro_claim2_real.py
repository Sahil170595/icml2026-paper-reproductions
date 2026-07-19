"""Claim 2 on REAL pretrained weights: Reynolds equivariant filtering (Eq. 6) +
Gaussian low-pass (Eq. 7) suppress the patch-grid artifact in DAVE attributions of
timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k on real ImageNet photos.

Stages (argv[1]):
  attr <i0> <i1>  - for images [i0,i1): compute effective-gradient attribution
                    A0 = g_eff * X (per-pixel), the Reynolds average over the exact
                    integer-translation group T = {-6,-3,0,3,6}^2 (25 transforms,
                    tau^-1 o W_L o tau via torch.roll), and the Gaussian low-pass
                    (sigma=4px) of the Reynolds map. Caches per-image tensors (.pt).
  metrics         - patch-lattice spectral energy fraction (energy at spatial
                    frequencies that are multiples of 14 cyc/img = the 16-px patch
                    lattice, DC excluded) for raw / Reynolds / Reynolds+low-pass;
                    position-locked artifact ||E_X[A]|| across images before/after;
                    writes results_real.json.
"""
import sys, os, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.dirname(HERE)
sys.path.insert(0, EP)
CACHE = os.path.join(EP, "_cache")
import vit_pretrained as V  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(1)
SHIFTS = [(dy, dx) for dy in (-6, -3, 0, 3, 6) for dx in (-6, -3, 0, 3, 6)]
N_IMG = 20


def eff_attr(m, X, target):
    Xf = X.clone().requires_grad_(True)
    y = m.forward(Xf[None], detach_op=True)[0, target]
    g, = torch.autograd.grad(y, Xf)
    return (g * X).sum(0)  # 224,224


def gauss_blur(a, sigma=4.0):
    r = int(3 * sigma)
    x = torch.arange(-r, r + 1, dtype=a.dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2)); k /= k.sum()
    k2 = torch.outer(k, k)
    pad = torch.nn.functional.pad(a[None, None], (r, r, r, r), mode="reflect")
    return torch.nn.functional.conv2d(pad, k2[None, None])[0, 0]


def stage_attr(i0, i1):
    m = V.ViT()
    paths = V.image_paths()[::3][:N_IMG]
    for i in range(i0, min(i1, len(paths))):
        p = paths[i]
        X = V.preprocess(p)
        w = os.path.basename(p).split("_")[0]
        t = V.WNID_IDX[w]
        A0 = eff_attr(m, X, t)
        acc = torch.zeros_like(A0)
        for dy, dx in SHIFTS:
            Xs = torch.roll(X, (dy, dx), dims=(1, 2))
            As = eff_attr(m, Xs, t)
            acc += torch.roll(As, (-dy, -dx), dims=(0, 1))
        Arey = acc / len(SHIFTS)
        Alow = gauss_blur(Arey.double()).float()
        torch.save(dict(A0=A0, Arey=Arey, Alow=Alow, target=t, img=os.path.basename(p)),
                   os.path.join(CACHE, "c2_attr_%02d.pt" % i))
        print(i, os.path.basename(p), "done")


def periodic_frac(a):
    """Fraction of AC energy in the exactly-16px-periodic subspace (orthogonal projection:
    average the map over the 14x14 grid of 16x16 tiles, tile it back)."""
    ac = a - a.mean()
    tile = ac.reshape(14, 16, 14, 16).mean((0, 2))
    tile = tile - tile.mean()
    return (196 * (tile ** 2).sum() / (ac ** 2).sum()).item()


def boundary_ratio(a):
    """Mean |finite difference| across patch boundaries / within-patch mean |difference|."""
    dcol = (a[:, 1:] - a[:, :-1]).abs()
    drow = (a[1:, :] - a[:-1, :]).abs()
    bmask = torch.zeros(223, dtype=torch.bool)
    bmask[15::16] = True
    b = torch.cat([dcol[:, bmask].flatten(), drow[bmask, :].flatten()]).mean()
    w = torch.cat([dcol[:, ~bmask].flatten(), drow[~bmask, :].flatten()]).mean()
    return (b / w).item()


def lattice_frac(a):
    """Fraction of spectral energy at 16-px lattice harmonics (freq multiple of 14, not 0)."""
    F = torch.fft.fft2(a - a.mean()).abs() ** 2
    n = a.shape[0]
    ky = torch.arange(n)[:, None].expand(n, n)
    kx = torch.arange(n)[None, :].expand(n, n)
    lat = (((ky % 14 == 0) & (ky != 0)) | ((kx % 14 == 0) & (kx != 0)))
    return (F[lat].sum() / F.sum()).item()


def stage_metrics():
    A0s, Areys, rows = [], [], []
    for i in range(N_IMG):
        d = torch.load(os.path.join(CACHE, "c2_attr_%02d.pt" % i))
        A0s.append(d["A0"]); Areys.append(d["Arey"])
        r = dict(img=d["img"])
        for tag, a in (("raw", d["A0"]), ("rey", d["Arey"]), ("low", d["Alow"])):
            r["per_" + tag] = periodic_frac(a)
            r["bnd_" + tag] = boundary_ratio(a)
            r["lat_" + tag] = lattice_frac(a)
        rows.append(r)
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    # position-locked artifact: 16px-periodic component of the ACROSS-IMAGE mean map
    def nrm(a): return a / (a.norm() + 1e-12)
    E0, Er = torch.stack([nrm(a) for a in A0s]).mean(0), torch.stack([nrm(a) for a in Areys]).mean(0)
    cors = [torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
            for a, b in zip(A0s, Areys)]
    res = dict(
        model="timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        n_images=n, group="integer translations {-6,-3,0,3,6}^2 (25), exact via roll",
        lowpass_sigma_px=4.0,
        periodic16_frac=dict(raw=mean("per_raw"), reynolds=mean("per_rey"), reynolds_lowpass=mean("per_low"),
                             drop_reynolds_pct=100 * (1 - mean("per_rey") / mean("per_raw")),
                             drop_full_pct=100 * (1 - mean("per_low") / mean("per_raw"))),
        boundary_ratio=dict(raw=mean("bnd_raw"), reynolds=mean("bnd_rey"), reynolds_lowpass=mean("bnd_low")),
        lattice_energy_frac=dict(raw=mean("lat_raw"), reynolds=mean("lat_rey"), reynolds_lowpass=mean("lat_low"),
                                 drop_full_pct=100 * (1 - mean("lat_low") / mean("lat_raw"))),
        position_locked_periodic=dict(raw=periodic_frac(E0), reynolds=periodic_frac(Er),
                                      drop_pct=100 * (1 - periodic_frac(Er) / periodic_frac(E0))),
        raw_vs_reynolds_corr=dict(mean=sum(cors) / len(cors), min=min(cors)),
        per_image=rows)
    json.dump(res, open(os.path.join(HERE, "results_real.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("periodic16_frac", "boundary_ratio", "lattice_energy_frac",
                                          "position_locked_periodic", "raw_vs_reynolds_corr")}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "attr":
        stage_attr(int(sys.argv[2]), int(sys.argv[3]))
    else:
        stage_metrics()
