"""Cross-platform re-verification of the staged evidence cache.

Re-runs a subset of the staged configs FROM SCRATCH (fresh _cache) with the
byte-identical engine scripts on a second platform, then compares every
recorded field against the bundle's _cache/*_done.json. Deterministic seeding
(numpy default_rng per-config) means the full recorded regret curves must
match if the cached evidence is genuine and reproducible.

Usage: python compare_verification.py <fresh_cache_dir> <bundle_cache_dir>
Writes verification.json next to this file and prints a per-field report.
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np

CONFIGS = ["c1_d10_s2p0", "c1_d100_s2p0", "c2_tshard_d20_T2000"]


def flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten(f"{prefix}[{i}]", v, out)
    else:
        out[prefix] = obj


def main():
    fresh, bundle = Path(sys.argv[1]), Path(sys.argv[2])
    report = {
        "platform_rerun": {
            "python": platform.python_version(),
            "system": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
        "configs": {},
    }
    all_ok = True
    for tag in CONFIGS:
        f = fresh / f"{tag}_done.json"
        b = bundle / f"{tag}_done.json"
        if not f.exists() or not b.exists():
            report["configs"][tag] = {"status": "missing", "fresh": f.exists(), "bundle": b.exists()}
            all_ok = False
            print(f"{tag}: MISSING (fresh={f.exists()} bundle={b.exists()})")
            continue
        fo, bo = {}, {}
        flatten("", json.loads(f.read_text()), fo)
        flatten("", json.loads(b.read_text()), bo)
        keys = sorted(set(fo) | set(bo))
        n_exact = n_close = 0
        worst_key, worst_rel = None, 0.0
        mismatches = []
        for k in keys:
            if k not in fo or k not in bo:
                mismatches.append({"field": k, "fresh": fo.get(k), "bundle": bo.get(k)})
                continue
            a, c = fo[k], bo[k]
            if a == c:
                n_exact += 1
            elif isinstance(a, float) and isinstance(c, float):
                rel = abs(a - c) / max(abs(c), 1e-12)
                if rel < 1e-9:
                    n_close += 1
                else:
                    mismatches.append({"field": k, "fresh": a, "bundle": c, "rel": rel})
                if rel > worst_rel:
                    worst_rel, worst_key = rel, k
            else:
                mismatches.append({"field": k, "fresh": a, "bundle": c})
        ok = not mismatches
        all_ok &= ok
        report["configs"][tag] = {
            "status": "exact_match" if (ok and n_close == 0) else ("match_1e-9" if ok else "MISMATCH"),
            "n_fields": len(keys), "n_exact": n_exact, "n_close_lt_1e-9": n_close,
            "worst_rel_diff": worst_rel, "worst_field": worst_key,
            "mismatches": mismatches[:20],
        }
        print(f"{tag}: {report['configs'][tag]['status']}  fields={len(keys)} exact={n_exact} "
              f"close={n_close} worst_rel={worst_rel:.3e} ({worst_key})")
    report["all_configs_reproduced"] = bool(all_ok)
    out = Path(__file__).resolve().parent / "verification.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"all_configs_reproduced={all_ok}  [written] {out.name}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
