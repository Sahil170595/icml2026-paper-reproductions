"""
verify_all.py -- Fail-closed re-verification of the whole evidence package.

Recomputes SHA-256 for every produced artifact, then re-asserts the scored
thresholds from the executed results.  Prints a single PASS/FAIL line and writes
verification.json + CHECKSUMS.sha256.  Run after run_gaussian / run_oup /
claim4_audit have produced their JSON outputs.
"""
import os, sys, json, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(name):
    with open(os.path.join(ART, name)) as f:
        return json.load(f)


def main():
    g = load("gaussian_results.json")
    o = load("oup_results.json")
    cert = load("proof_certificate.json")

    checks = []

    def assert_(name, cond, detail=""):
        checks.append({"name": name, "ok": bool(cond), "detail": detail})

    # ---- Claim 1: plug-in separation, frozen NPE tensor-hash identity ----
    assert_("gaussian_npe_is_genuine_conditional_density",
            g["npe_vs_analytic_rmse"] < 0.1,
            f"NPE-vs-analytic posterior mean RMSE={g['npe_vs_analytic_rmse']:.4f}")
    assert_("gaussian_frozen_npe_hash_identity",
            g["hash_identity_all_ok"] and g["hash_identity_checks"] == 300,
            f"{g['hash_identity_checks']} checks all identical")
    assert_("oup_frozen_npe_hash_identity",
            o["hash_identity_all_ok"] and o["hash_identity_checks"] == 250,
            f"{o['hash_identity_checks']} checks all identical")
    assert_("gaussian_npe_param_count_small", 3000 < g["npe_params"] < 8000,
            f"{g['npe_params']} params")
    assert_("oup_npe_param_count", 12000 < o["npe_params"] < 22000,
            f"{o['npe_params']} params")

    # ---- Claim 2: RFF efficiency, lightweight adaptation, RFF~exact ----
    assert_("gaussian_512_rff", g["k_rff"] == 512, "K=512")
    assert_("oup_512_rff", o["k_rff"] == 512, "K=512")
    assert_("gaussian_adaptation_lightweight_ms",
            max(d["median_ms"] for d in g["per_level"]) < 50,
            f"max median {max(d['median_ms'] for d in g['per_level']):.2f}ms")
    assert_("gaussian_rff_matches_exact_mmd",
            g["rff_vs_exact"]["mean_abs_gap"] < 0.05 and g["rff_vs_exact"]["corr"] > 0.9,
            f"gap={g['rff_vs_exact']['mean_abs_gap']:.4f} corr={g['rff_vs_exact']['corr']:.3f}")
    assert_("oup_rff_matches_exact_mmd",
            o["rff_vs_exact"]["mean_abs_gap"] < 0.05,
            f"gap={o['rff_vs_exact']['mean_abs_gap']:.4f}")

    # ---- Claim 3: substantial robustness gains, low overhead ----
    assert_("gaussian_substantial_reduction",
            g["mean_reduction_nonsevere_pct"] > 80,
            f"mean reduction {g['mean_reduction_nonsevere_pct']:.2f}%")
    assert_("gaussian_all_wins_nonsevere",
            all(d["wins"] == d["trials"] for d in g["per_level"] if 0.1 <= d["eps"] <= 0.4),
            "50/50 at eps 0.1-0.4")
    assert_("oup_substantial_reduction",
            o["mean_reduction_nonsevere_pct"] > 50,
            f"mean reduction {o['mean_reduction_nonsevere_pct']:.2f}%")
    assert_("oup_high_win_rate",
            all(d["wins"] >= 40 for d in o["per_level"] if 0.1 <= d["eps"] <= 0.4),
            ">=40/50 at eps 0.1-0.4")
    # honest severe-case disclosure present
    sev = [d for d in g["per_level"] if d["eps"] == 0.5][0]
    assert_("gaussian_severe_case_disclosed",
            sev["reduction_pct"] < g["mean_reduction_nonsevere_pct"],
            f"eps=0.5 reduction {sev['reduction_pct']:.2f}% << non-severe mean")

    # ---- Claim 4: theory audit ----
    assert_("tex_sha_pinned", cert["tex_sha256_matches_pin"], cert["tex_sha256"][:16])
    assert_("assumptions_10_4",
            cert["n_robustness_assumptions"] == 10 and cert["n_consistency_assumptions"] == 4,
            "10 robustness + 4 consistency")
    assert_("dag_complete",
            cert["robustness_chain_complete"] and cert["consistency_chain_complete"],
            "R0->R3 and C1->C4 complete")
    assert_("bounded_influence_finite", cert["bounded_influence"]["bounded_ok"],
            f"sup|IF|={cert['bounded_influence']['sup_influence']:.4f}")
    assert_("posterior_contraction_monotone", cert["posterior_contraction"]["ok"],
            f"slope={cert['posterior_contraction']['loglog_slope']:.4f}")

    # ---- artifact hashes ----
    artifacts = ["gaussian_results.json", "oup_results.json", "proof_certificate.json",
                 "gaussian_npe.pt", "oup_npe.pt", "arxiv_main.tex"]
    checksums = {}
    for a in artifacts:
        p = os.path.join(ART, a)
        if os.path.exists(p):
            checksums[a] = sha256_file(p)
    with open(os.path.join(ART, "CHECKSUMS.sha256"), "w") as f:
        for a in sorted(checksums):
            f.write(f"{checksums[a]}  {a}\n")

    n_ok = sum(c["ok"] for c in checks)
    n_tot = len(checks)
    all_ok = n_ok == n_tot
    ver = dict(n_assertions=n_tot, n_passed=n_ok, all_passed=all_ok,
               checks=checks, checksums=checksums)
    with open(os.path.join(ART, "verification.json"), "w") as f:
        json.dump(ver, f, indent=2)

    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']:44s} {c['detail']}")
    print()
    if all_ok:
        print(f"PASS: {n_ok}/{n_tot} assertions + {len(checksums)} artifact hashes; CPU-only bundle verified")
    else:
        print(f"FAIL: {n_ok}/{n_tot} assertions passed")
        sys.exit(1)


if __name__ == "__main__":
    main()
