import os
import tempfile
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


# -------------------- Global params --------------------
LTPD_TOTAL = 0.01
#LIFE_YEARS = 1
#LTPD_ANNUAL = LTPD_TOTAL / LIFE_YEARS

conf_level = 0.90
hR = 0.02
N_max = 6000

r_allow = 0  # acceptance number C=0
def pass_rule_test(x_fail: int) -> bool:
    return x_fail <= r_allow

N_REP = 100
SEED_BASE = 20250101

HIST_CLASS_NAMES = ["2020", "2021", "2022", "2023", "2024"]
HIST_CLASS_DPPM = np.array([10000, 5000, 2000, 1000, 200], dtype=float)

SEG_K = 5
SEG_FACTOR = np.array([1.30, 1.15, 1.00, 0.85, 0.70], dtype=float)
SEG_FACTOR = SEG_FACTOR / SEG_FACTOR.mean()

NEW_PROXY_CLASS = "2024"
NEW_PROXY_SEG = 5

# 2025 pool (true simulation ppm)
NEW_2025_DPPM = 15000
NEW_p_true_2025 = NEW_2025_DPPM / 1e6
Y25_N_BATCHES_POOL = 10
Y25_PER_BATCH_POOL = 4000
TEST_SAMPLE_REPLACE_IF_NEEDED = True

HX_PPM = 2000

# --------- Online alpha control (DIRECT) ----------
alpha_init = 0.999
ALPHA_MIN = 0.001
ALPHA_MAX = 0.999

STEP_SIZE = 1.0
DELTA_MAX = 0.01 #0.05

WIN = 10

# error scale
ERR_SCALE = max(NEW_p_true_2025, 1e-6)

# Fixed split
BUILD_FRAC = 0.50
SPLIT_SEED_FIXED = SEED_BASE + 55555

HIST_PER_SEG_BATCH = 4000
N_PER_SEG_BATCH = 1  # one batch per segment

P_BAD = LTPD_TOTAL
CONF_TARGET = 0.90

QHAT_ONLY_TIGHTEN = True
XNEW_TIMES_10 = True


# -------------------- Utils --------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def clamp_alpha(a: float) -> float:
    return max(ALPHA_MIN, min(ALPHA_MAX, float(a)))

def clamp_step(d: float) -> float:
    return max(-DELTA_MAX, min(DELTA_MAX, float(d)))


def kde1d_density(R_grid: np.ndarray, R_i: np.ndarray, hR: float) -> np.ndarray:
    """
    Unweighted KDE over R_grid.
    """
    delta = R_grid[1] - R_grid[0]
    if R_i.size == 0:
        dens = np.ones_like(R_grid, dtype=float)
        return dens / (dens.sum() * delta)

    # Gaussian kernel: dnorm((R-Ri)/hR)/hR
    # Vectorized: for each grid point, sum over i
    z = (R_grid[:, None] - R_i[None, :]) / hR
    dens = norm.pdf(z) / hR
    dens = dens.sum(axis=1)
    dens = np.maximum(dens, 0.0)
    dens = dens / (dens.sum() * delta)
    return dens


def kde1d_density_weighted(R_grid: np.ndarray, R_i: np.ndarray, w_i: np.ndarray, hR: float) -> np.ndarray:
    """
    Weighted KDE over R_grid, with safeguard sum(w)<=0 -> uniform weights.
    """
    delta = R_grid[1] - R_grid[0]
    if R_i.size == 0:
        dens = np.ones_like(R_grid, dtype=float)
        return dens / (dens.sum() * delta)

    w = np.maximum(w_i.astype(float), 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)

    z = (R_grid[:, None] - R_i[None, :]) / hR
    dens = (w[None, :] * (norm.pdf(z) / hR)).sum(axis=1)
    dens = np.maximum(dens, 0.0)
    dens = dens / (dens.sum() * delta)
    return dens


def posterior_conf_good(N: int, prior_dens: np.ndarray, R_grid: np.ndarray, p_bad_eff: float) -> float:
    """
    Conf0(N) = Pr(R >= 1 - p_bad_eff | r=0, N, X) under posterior proportional to prior(R)*R^N.
    """
    if N is None or N <= 0:
        return np.nan
    delta = R_grid[1] - R_grid[0]
    like = np.power(R_grid, N)
    post = prior_dens * like
    Z = post.sum() * delta
    if Z <= 0:
        return np.nan
    R_thr = 1.0 - p_bad_eff
    idx = (R_grid >= R_thr)
    return (post[idx].sum() * delta) / Z


def solve_Nstar_by_postconf(prior_dens: np.ndarray, R_grid: np.ndarray, qhat: float,
                           p_bad: float, conf_target: float, N_max: int) -> int | None:
    p_bad_eff = clamp01(p_bad - qhat)
    for N in range(1, N_max + 1):
        confN = posterior_conf_good(N, prior_dens, R_grid, p_bad_eff)
        if np.isnan(confN):
            continue
        if confN >= conf_target:
            return N
    return None


def prior_pass_prob(N: int, prior_dens: np.ndarray, R_grid: np.ndarray) -> float:
    """
    E_pi[R^N] approximated on grid.
    """
    if N is None or N <= 0:
        return np.nan
    delta = R_grid[1] - R_grid[0]
    out = (np.power(R_grid, N) * prior_dens).sum() * delta
    return float(np.clip(out, 0.0, 1.0))


def p_prior_implied(N: int, pass_prob: float) -> float:
    """
    implied p such that (1-p)^N = pass_prob
    """
    if N is None or N <= 0 or np.isnan(pass_prob):
        return np.nan
    pp = float(np.clip(pass_prob, 0.0, 1.0))
    return 1.0 - (pp ** (1.0 / N))


def build_fixed_ppm_table() -> pd.DataFrame:
    base_seq = np.repeat(HIST_CLASS_DPPM, SEG_K) * np.tile(SEG_FACTOR, len(HIST_CLASS_NAMES))
    ppm = np.round(base_seq).astype(int)

    # enforce strictly decreasing overall (matches your R loop)
    for i in range(1, len(ppm)):
        if ppm[i] >= ppm[i - 1]:
            ppm[i] = ppm[i - 1] - 1
    ppm = np.maximum(ppm, 1)

    yrs = np.repeat(HIST_CLASS_NAMES, SEG_K)
    seg = np.tile(np.arange(1, SEG_K + 1), len(HIST_CLASS_NAMES))

    df = pd.DataFrame({"类别": yrs, "段": seg, "X_ppm": ppm})
    df = df.sort_values(["类别", "段"]).reset_index(drop=True)
    return df


def simulate_hist_2020_2024(seed: int, ppm_table: pd.DataFrame,
                           per_batch: int = 4000, n_per_seg_batch: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cls in HIST_CLASS_NAMES:
        for s in range(1, SEG_K + 1):
            seg_ppm = int(ppm_table.loc[(ppm_table["类别"] == cls) & (ppm_table["段"] == s), "X_ppm"].iloc[0])
            p_fail = seg_ppm / 1e6
            for b in range(1, n_per_seg_batch + 1):
                fail_vec = rng.binomial(1, p_fail, size=per_batch)
                rows.append(pd.DataFrame({
                    "批次编号": [f"{cls}_S{s}_L{b}"] * per_batch,
                    "类别": [cls] * per_batch,
                    "段": [s] * per_batch,
                    "X_ppm": [seg_ppm] * per_batch,
                    "是否失效": fail_vec.astype(int)
                }))
    return pd.concat(rows, ignore_index=True)


def simulate_2025_pool(seed: int, n_batches: int, per_batch: int, p_true: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    total = n_batches * per_batch
    fail = rng.binomial(1, p_true, size=total).astype(int)
    batch_id = np.repeat(np.arange(1, n_batches + 1), per_batch)
    return pd.DataFrame({"批次编号": batch_id, "类别": "2025", "是否失效": fail})


def summarise_batches(df_hist: pd.DataFrame) -> pd.DataFrame:
    g = df_hist.groupby(["批次编号", "类别", "段"], as_index=False).agg(
        n=("是否失效", "size"),
        r=("是否失效", "sum"),
        X_ppm=("X_ppm", "first")
    )
    g["R_hat"] = np.clip(1.0 - g["r"] / g["n"], 0.0, 1.0)
    g["p_hat_batch"] = np.clip(g["r"] / g["n"], 0.0, 1.0)
    g["p_true_local"] = g["X_ppm"] / 1e6
    return g


def prior_hist_given_x(R_grid: np.ndarray, batch_tbl: pd.DataFrame,
                       x_ppm: int, hX: float, hR: float) -> np.ndarray:
    w = norm.pdf((x_ppm - batch_tbl["X_ppm"].to_numpy(dtype=float)) / hX)
    return kde1d_density_weighted(R_grid, batch_tbl["R_hat"].to_numpy(dtype=float), w, hR)


def prior_new_from_proxy(R_grid: np.ndarray, batch_tbl: pd.DataFrame, hR: float) -> np.ndarray:
    tbl_new = batch_tbl[(batch_tbl["类别"] == NEW_PROXY_CLASS) & (batch_tbl["段"] == NEW_PROXY_SEG)]
    if tbl_new.shape[0] == 0:
        delta = R_grid[1] - R_grid[0]
        dens = np.ones_like(R_grid, dtype=float)
        return dens / (dens.sum() * delta)
    return kde1d_density(R_grid, tbl_new["R_hat"].to_numpy(dtype=float), hR)


def mix_prior(prior_hist_x: np.ndarray, prior_new: np.ndarray, R_grid: np.ndarray, alpha: float) -> np.ndarray:
    delta = R_grid[1] - R_grid[0]
    prior = alpha * prior_hist_x + (1.0 - alpha) * prior_new
    prior = np.maximum(prior, 0.0)
    prior = prior / (prior.sum() * delta)
    return prior


def estimate_qhat_weighted_split_fixed(batch_tbl_fixed: pd.DataFrame,
                                       build_ids: np.ndarray, calib_ids: np.ndarray,
                                       alpha_now: float, R_grid: np.ndarray,
                                       hX: float, hR: float, conf_level: float) -> float:
    build = batch_tbl_fixed.iloc[build_ids].copy()
    calib = batch_tbl_fixed.iloc[calib_ids].copy()
    if calib.shape[0] == 0:
        return 0.0

    prior_new_build = prior_new_from_proxy(R_grid, build, hR)
    delta = R_grid[1] - R_grid[0]

    phat = np.zeros(calib.shape[0], dtype=float)
    ptrue = calib["p_true_local"].to_numpy(dtype=float)

    for i in range(calib.shape[0]):
        x_i = int(calib["X_ppm"].iloc[i])
        prior_hist_xi = prior_hist_given_x(R_grid, build, x_i, hX=hX, hR=hR)
        prior_xi = mix_prior(prior_hist_xi, prior_new_build, R_grid, alpha=alpha_now)
        ER = (R_grid * prior_xi).sum() * delta
        phat[i] = 1.0 - ER

    resid = ptrue - phat
    qhat = float(np.quantile(resid, conf_level, method="linear"))  
    return qhat


def pick_writable_dir() -> str:
    cand1 = os.path.join(os.path.expanduser("~"), "output_A")
    try:
        os.makedirs(cand1, exist_ok=True)
        # check writable
        testfile = os.path.join(cand1, ".write_test")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
        return cand1
    except Exception:
        cand2 = os.path.join(tempfile.gettempdir(), "output_A")
        os.makedirs(cand2, exist_ok=True)
        return cand2


ppm_table_fixed = build_fixed_ppm_table()
R_grid = np.arange(0.0, 1.0001, 0.001)

seed_hist_fixed = SEED_BASE + 1
df_hist_fixed = simulate_hist_2020_2024(seed_hist_fixed, ppm_table_fixed, HIST_PER_SEG_BATCH, N_PER_SEG_BATCH)
batch_tbl_fixed = summarise_batches(df_hist_fixed)

# proxy x_new used to build fixed prior components
x_new_ppm = int(ppm_table_fixed.loc[(ppm_table_fixed["类别"] == NEW_PROXY_CLASS) &
                                   (ppm_table_fixed["段"] == NEW_PROXY_SEG), "X_ppm"].iloc[0])
if XNEW_TIMES_10:
    x_new_ppm = int(10 * x_new_ppm)  # used ppm for weighting
prior_hist_xnew_fixed = prior_hist_given_x(R_grid, batch_tbl_fixed, x_new_ppm, hX=HX_PPM, hR=hR)
prior_new_fixed = prior_new_from_proxy(R_grid, batch_tbl_fixed, hR=hR)

rng_split = np.random.default_rng(SPLIT_SEED_FIXED)
id_all = rng_split.permutation(batch_tbl_fixed.shape[0])
n_build = max(1, int(np.floor(BUILD_FRAC * batch_tbl_fixed.shape[0])))
build_ids = id_all[:n_build]
calib_ids = id_all[n_build:]
if calib_ids.size == 0:
    raise RuntimeError("Calibration split empty. Reduce BUILD_FRAC or increase data.")

# =========================================================
# Loop
# =========================================================
rep_rows = []
alpha_t = alpha_init

err_hist = []  # store err history for rolling mean

x_new_ppm_base = int(x_new_ppm / 10) if XNEW_TIMES_10 else int(x_new_ppm)
p_prior_fixed_from_xnew = (x_new_ppm_base / 1e6) #NEW_p_true_2025/10

safe_out_dir = pick_writable_dir()
print(f"\nFiles will be saved to: {safe_out_dir}\n")

for t in range(1, N_REP + 1):
    seed_2025 = SEED_BASE + 999999 + t
    alpha_before = alpha_t

    prior_dens_xnew = mix_prior(prior_hist_xnew_fixed, prior_new_fixed, R_grid, alpha=alpha_before)

    qhat_raw = estimate_qhat_weighted_split_fixed(
        batch_tbl_fixed, build_ids, calib_ids,
        alpha_now=alpha_before,
        R_grid=R_grid, hX=HX_PPM, hR=hR, conf_level=conf_level
    )
    qhat_use = max(qhat_raw, 0.0) if QHAT_ONLY_TIGHTEN else qhat_raw

    N_star = solve_Nstar_by_postconf(prior_dens_xnew, R_grid, qhat_use, P_BAD, CONF_TARGET, N_max)

    df_2025_pool = simulate_2025_pool(seed_2025, Y25_N_BATCHES_POOL, Y25_PER_BATCH_POOL, NEW_p_true_2025)
    test_n = df_2025_pool.shape[0]

    replace_flag = False
    if N_star is not None and N_star > test_n:
        if TEST_SAMPLE_REPLACE_IF_NEEDED:
            replace_flag = True
        else:
            N_star = None

    F_t = None
    p_obs_t = np.nan
    passed = None

    if N_star is not None and N_star > 0:
        rng = np.random.default_rng(seed_2025 + 777)  # separate stream
        idx = rng.choice(test_n, size=N_star, replace=replace_flag)
        F_t = int(df_2025_pool["是否失效"].to_numpy()[idx].sum())
        p_obs_t = F_t / N_star
        passed = int(pass_rule_test(F_t))


    pass_prob = prior_pass_prob(N_star, prior_dens_xnew, R_grid) if N_star is not None else np.nan
    p_prior_imp = p_prior_implied(N_star, pass_prob) if N_star is not None else np.nan

    # NEW REQUIREMENT (user-specific: *10 means all life)
    p_prior_t = p_prior_fixed_from_xnew * 10.0
    err_t = (p_obs_t - p_prior_t) if not np.isnan(p_obs_t) else np.nan

    if not np.isnan(err_t):
        err_hist.append(err_t)

    err_smooth = np.nan
    if len(err_hist) > 0:
        D = min(WIN, len(err_hist))
        err_smooth = float(np.mean(err_hist[-D:]))
        delta_alpha = clamp_step(STEP_SIZE * (err_smooth / ERR_SCALE))
        alpha_t = clamp_alpha(alpha_t - delta_alpha)

    alpha_after = alpha_t

    rep_rows.append({
        "rep_id": t,
        "alpha_before": alpha_before,
        "alpha_after": alpha_after,
        "N_star": N_star,
        "F_t": F_t,
        "p_obs_t": p_obs_t,
        "p_prior_imp": p_prior_imp,
        "p_prior_xnew": p_prior_t,
        "err_t": err_t,
        "err_smooth": err_smooth,
        "pass": passed,
        "qhat_raw": qhat_raw,
        "qhat_used": qhat_use,
        "x_new_ppm": x_new_ppm,
        "x_new_ppm_base": x_new_ppm_base,
    })

    if t % 10 == 0:
        print(f"t={t} alpha={alpha_before:.3f}->{alpha_after:.3f} "
              f"N*={N_star} pass={passed} "
              f"p_obs={p_obs_t if not np.isnan(p_obs_t) else 'NA'} "
              f"p_prior_xnew={p_prior_t:.6g} "
              f"err={err_t if not np.isnan(err_t) else 'NA'} "
              f"WinMean={err_smooth if not np.isnan(err_smooth) else 'NA'}")


rep_df = pd.DataFrame(rep_rows)

sum_df = pd.DataFrame([{
    "N_rep": rep_df.shape[0],
    "pass_rate": float(rep_df["pass"].dropna().mean()) if rep_df["pass"].notna().any() else np.nan,
    "alpha_start": float(rep_df["alpha_before"].iloc[0]),
    "alpha_end": float(rep_df["alpha_after"].iloc[-1]),
    "Nstar_mean": float(rep_df["N_star"].dropna().mean()) if rep_df["N_star"].notna().any() else np.nan,
    "Nstar_min": int(rep_df["N_star"].dropna().min()) if rep_df["N_star"].notna().any() else None,
    "Nstar_max": int(rep_df["N_star"].dropna().max()) if rep_df["N_star"].notna().any() else None,
}])

print("\n===== SUMMARY =====")
print(sum_df)

# -------------------- Save --------------------
fn_rep = os.path.join(safe_out_dir, f"A_direct_scheme1_2025ppm_20250101{NEW_2025_DPPM}{N_REP}.csv")
fn_sum = os.path.join(safe_out_dir, f"A_direct_scheme1_2025ppm_20250101{NEW_2025_DPPM}{N_REP}_summary.csv")
rep_df.to_csv(fn_rep, index=False)
sum_df.to_csv(fn_sum, index=False)

# plots
plt.figure()
plt.plot(rep_df["rep_id"], rep_df["N_star"])
plt.xlabel("Iteration t")
plt.ylabel("N_star")
plt.title(f"N_star trend (direct update, alpha starts near 1, 2025ppm={NEW_2025_DPPM})")
plt.tight_layout()
plt.savefig(os.path.join(safe_out_dir, f"Nstar_trend_2025ppm_20250101{NEW_2025_DPPM}{N_REP}.png"), dpi=150)
plt.close()

plt.figure()
plt.plot(rep_df["rep_id"], rep_df["alpha_before"])
plt.xlabel("Iteration t")
plt.ylabel("alpha_before")
plt.title("alpha trend (before update)")
plt.tight_layout()
plt.savefig(os.path.join(safe_out_dir, f"alpha_trend_2025ppm_20250101{NEW_2025_DPPM}{N_REP}.png"), dpi=150)
plt.close()


# =========================================================
# Confidence Level Table (r=0, fixed N)
# =========================================================
XNEW_PPM_GRID = [20, 80, 140, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]

t_anchor = 1
N_fixed = int(rep_df.loc[rep_df["rep_id"] == t_anchor, "N_star"].iloc[0])
alpha_anchor = float(rep_df.loc[rep_df["rep_id"] == t_anchor, "alpha_before"].iloc[0])
qhat_anchor = float(rep_df.loc[rep_df["rep_id"] == t_anchor, "qhat_used"].iloc[0])
p_bad_eff_anchor = clamp01(P_BAD - qhat_anchor)

def calc_conf0_given_xnew(x_new_ppm_input: int) -> dict:
    x_used = int(round(x_new_ppm_input))
    if XNEW_TIMES_10:
        x_used = int(round(10 * x_used))

    prior_hist_x = prior_hist_given_x(R_grid, batch_tbl_fixed, x_used, hX=HX_PPM, hR=hR)
    prior_dens_x = mix_prior(prior_hist_x, prior_new_fixed, R_grid, alpha=alpha_anchor)

    conf0 = posterior_conf_good(N_fixed, prior_dens_x, R_grid, p_bad_eff_anchor)

    pass_prob_x = prior_pass_prob(N_fixed, prior_dens_x, R_grid)
    p_imp_x = p_prior_implied(N_fixed, pass_prob_x)
    ppm_imp_x = 1e6 * p_imp_x

    delta = R_grid[1] - R_grid[0]
    ER_x = (R_grid * prior_dens_x).sum() * delta
    ppm_mean_x = 1e6 * (1.0 - ER_x)

    return {
        "x_new_ppm_input": x_new_ppm_input,
        "x_new_ppm_used": x_used,
        "N_fixed": N_fixed,
        "alpha_anchor": alpha_anchor,
        "qhat_anchor": qhat_anchor,
        "p_bad_eff": p_bad_eff_anchor,
        "Conf0_r0": conf0,
        "ppm_implied_at_N": ppm_imp_x,
        "ppm_mean_prior": ppm_mean_x
    }

conf_tbl = pd.DataFrame([calc_conf0_given_xnew(x) for x in XNEW_PPM_GRID])
print("\n===== Our-method Confidence Level Table (r=0, fixed N) =====")
print(conf_tbl)

fn_conf_tbl = os.path.join(
    safe_out_dir,
    f"A_confTable_r0_fixedN_from_t_20250101{t_anchor}_2025ppm{NEW_2025_DPPM}_NREP{N_REP}.csv"
)
conf_tbl.to_csv(fn_conf_tbl, index=False)

plt.figure()
plt.plot(conf_tbl["x_new_ppm_input"], conf_tbl["Conf0_r0"], marker="o")
plt.axhline(CONF_TARGET, linestyle="--")
plt.xlabel("X_new (ppm input)")
plt.ylabel("Confidence level: Pr(p<=p_bad_eff | r=0, N_fixed, X_new)")
plt.title(f"Our-method confidence vs X_new (r=0, N_fixed={N_fixed}, alpha={alpha_anchor:.3f}, qhat={qhat_anchor:.4g})")
plt.tight_layout()
fn_conf_png = os.path.join(
    safe_out_dir,
    f"A_confCurve_r0_fixedN_from_t_20250101{t_anchor}_2025ppm{NEW_2025_DPPM}_NREP{N_REP}.png"
)
plt.savefig(fn_conf_png, dpi=150)
plt.close()

print("\nSaved:")
print(" rep_df:", fn_rep)
print(" sum_df:", fn_sum)
print(" conf_tbl:", fn_conf_tbl)
print(" conf_png:", fn_conf_png)
