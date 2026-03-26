# 导入系统路径与临时文件目录相关模块
import os

# 导入创建临时输出目录时会用到的模块
import tempfile

# 导入数值计算库，用于数组、随机数、向量化运算
import numpy as np

# 导入表格处理库，用于读写 Excel / CSV 和 DataFrame 操作
import pandas as pd

# 从 scipy.stats 导入正态分布对象，用于 Gaussian kernel 权重和密度计算
from scipy.stats import norm

# 导入画图库，用于输出 N* 趋势图等
import matplotlib.pyplot as plt


# =========================================================
# Global params
# =========================================================

# LTPD_TOTAL: 目标坏品率阈值（这里用总失效率口径）
# 例如 0.01 表示 1% 的坏品率门槛
LTPD_TOTAL = 0.01

# conf_level: conformal 校准时使用的分位数水平
# 例如 0.90 表示取残差的 90% 分位数
conf_level = 0.90

# hR: R 方向 KDE 的带宽
# R = 可靠度 = 1 - p
# 该参数控制对历史 R_hat 做核密度平滑时的平滑程度
hR = 0.02

# N_max: 求解 N_star 时搜索的最大样本量上限
# 程序会从 N=1 一直搜索到 N_max，找到第一个满足 posterior confidence >= CONF_TARGET 的 N
N_max = 6000

# r_allow: acceptance number
# 当前设置为 0，表示“零失效通过”规则
r_allow = 0

# pass_rule_test: 给定观测到的失效数 x_fail，判断是否通过
def pass_rule_test(x_fail: int) -> bool:
    # 若失效数不超过允许值，则返回 True
    return x_fail <= r_allow

# N_REP: 在线更新 alpha 的重复实验次数
N_REP = 100

# SEED_BASE: 基础随机种子，方便复现实验结果
SEED_BASE = 20250101

# HIST_CLASS_NAMES: 历史类别名称
# 在模拟模式下，对应不同年份
HIST_CLASS_NAMES = ["2020", "2021", "2022", "2023", "2024"]

# HIST_CLASS_DPPM: 各历史类别对应的基础 DPPM 水平
# 模拟模式下用它生成历史 ppm 表
HIST_CLASS_DPPM = np.array([10000, 5000, 2000, 1000, 200], dtype=float)

# SEG_K: 每个历史类别再细分成多少段
SEG_K = 5

# SEG_FACTOR: 每段相对基础 ppm 的倍率
# 用于构造一个类别内部逐段变化的 ppm 水平
SEG_FACTOR = np.array([1.30, 1.15, 1.00, 0.85, 0.70], dtype=float)

# 对 SEG_FACTOR 做均值归一化，避免整体水平偏离原始 HIST_CLASS_DPPM
SEG_FACTOR = SEG_FACTOR / SEG_FACTOR.mean()

# NEW_PROXY_CLASS: 用作“接近当前新芯片”的 proxy 类别
# 在当前设置中使用 2024 作为 proxy
NEW_PROXY_CLASS = "2024"

# NEW_PROXY_SEG: proxy 类别中的哪一段用于构造 prior_new
# 公司格式表里当前默认没有“段”的细分，所以统一设为 1
NEW_PROXY_SEG = 1

# NEW_2025_DPPM: 2025 测试池的真实 DPPM（模拟时作为 ground truth）
NEW_2025_DPPM = 140

# NEW_p_true_2025: 将 2025 的 DPPM 转成失效率 p
NEW_p_true_2025 = NEW_2025_DPPM / 1e6

# Y25_N_BATCHES_POOL: 2025 模拟池中批次数
Y25_N_BATCHES_POOL = 10

# Y25_PER_BATCH_POOL: 每个 2025 批次中的样本数
Y25_PER_BATCH_POOL = 4000

# TEST_SAMPLE_REPLACE_IF_NEEDED:
# 如果 N_star 比测试池样本数还大，是否允许有放回抽样
TEST_SAMPLE_REPLACE_IF_NEEDED = True

# HX_PPM: X 方向（ppm 方向）Gaussian kernel 的带宽
# 用于历史批次与目标 x_ppm 的相似性加权
HX_PPM = 2000

# alpha_init: alpha 的初始值
# alpha 越大，越偏向历史先验；越小，越偏向 new/proxy prior
alpha_init = 0.999

# ALPHA_MIN: alpha 的下界，防止更新后过小
ALPHA_MIN = 0.001

# ALPHA_MAX: alpha 的上界，防止更新后超过 1
ALPHA_MAX = 0.999

# STEP_SIZE: alpha 更新时的基础步长
STEP_SIZE = 1.0

# DELTA_MAX: 每一步 alpha 的最大允许变化幅度
DELTA_MAX = 0.01

# WIN: err_smooth 的窗口长度
# 例如 WIN=10 表示对最近 10 次 err_t 做滚动平均
WIN = 10

# ERR_SCALE: 对误差归一化的尺度
# 避免 err_t 很小或很大时导致 alpha 更新过激
ERR_SCALE = max(NEW_p_true_2025, 1e-6)

# BUILD_FRAC: 固定 split 中 build 集所占比例
BUILD_FRAC = 0.50

# SPLIT_SEED_FIXED: 固定 split 的随机种子
SPLIT_SEED_FIXED = SEED_BASE + 55555

# HIST_PER_SEG_BATCH: 模拟历史数据时，每个段每批次的样本量
HIST_PER_SEG_BATCH = 4000

# N_PER_SEG_BATCH: 每个段生成几个批次
N_PER_SEG_BATCH = 1

# P_BAD: 坏品率阈值
# 当前直接用 LTPD_TOTAL
P_BAD = LTPD_TOTAL

# CONF_TARGET: 求解 N_star 时要求达到的 posterior confidence 阈值
CONF_TARGET = 0.90

# QHAT_ONLY_TIGHTEN:
# 若为 True，则 qhat 只允许往“更保守”方向修正，即 qhat_use = max(qhat_raw, 0)
QHAT_ONLY_TIGHTEN = True

# XNEW_TIMES_10:
# 若为 True，则某些表格/计算中将 x_new 放大为 10 倍（对应 all-life 口径）
XNEW_TIMES_10 = True


# =========================================================
# Mode switches
# =========================================================

# GENERATE_SIM_COMPANY_INPUT:
# 若为 True，则先用模拟历史数据自动生成一份“公司格式”的输入表
# 这样即使还没有真实公司数据，也可以先用相同格式测试主程序
GENERATE_SIM_COMPANY_INPUT = True

# USE_REAL_INPUT:
# 若为 True，则主程序从公司格式表中读取历史数据
# 若为 False，则仍走旧的模拟输入路径
USE_REAL_INPUT = True

# REAL_INPUT_PATH:
# 公司格式输入表的路径
# 若 GENERATE_SIM_COMPANY_INPUT=True，则模拟数据会先导出到这个路径
REAL_INPUT_PATH = r"/Users/cezhang/Desktop/programA_py/company_input_from_simulation.xlsx"

# REAL_INPUT_SHEET:
# 若输入文件是 Excel，则读取哪个 sheet
REAL_INPUT_SHEET = 0


# =========================================================
# Utils
# =========================================================

# clamp01: 将一个数裁剪到 [0,1] 区间
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

# clamp_alpha: 将 alpha 裁剪到 [ALPHA_MIN, ALPHA_MAX] 区间
def clamp_alpha(a: float) -> float:
    return max(ALPHA_MIN, min(ALPHA_MAX, float(a)))

# clamp_step: 将一次 alpha 更新步长裁剪到 [-DELTA_MAX, DELTA_MAX]
def clamp_step(d: float) -> float:
    return max(-DELTA_MAX, min(DELTA_MAX, float(d)))

# kde1d_density:
# 在 R_grid 上对一组 R_i 做无权重 KDE
# 这里使用 Gaussian kernel
def kde1d_density(R_grid: np.ndarray, R_i: np.ndarray, hR: float) -> np.ndarray:
    # delta: R_grid 的网格步长，用于数值积分归一化
    delta = R_grid[1] - R_grid[0]

    # 如果没有输入样本，则退化成均匀密度
    if R_i.size == 0:
        dens = np.ones_like(R_grid, dtype=float)
        return dens / (dens.sum() * delta)

    # z: 标准化距离矩阵，形状为 [len(R_grid), len(R_i)]
    z = (R_grid[:, None] - R_i[None, :]) / hR

    # norm.pdf(z)/hR: Gaussian kernel 密度
    dens = norm.pdf(z) / hR

    # 对所有输入点求和，得到每个 R_grid 点上的总密度
    dens = dens.sum(axis=1)

    # 防止出现负数（理论上不会，这里作为数值保护）
    dens = np.maximum(dens, 0.0)

    # 归一化，使其在网格上积分为 1
    dens = dens / (dens.sum() * delta)

    return dens

# kde1d_density_weighted:
# 在 R_grid 上对一组 R_i 做带权 KDE
# w_i 是每个历史批次的权重
def kde1d_density_weighted(R_grid: np.ndarray, R_i: np.ndarray, w_i: np.ndarray, hR: float) -> np.ndarray:
    # delta: 网格步长
    delta = R_grid[1] - R_grid[0]

    # 若输入为空，则退化成均匀密度
    if R_i.size == 0:
        dens = np.ones_like(R_grid, dtype=float)
        return dens / (dens.sum() * delta)

    # 将权重裁剪为非负
    w = np.maximum(w_i.astype(float), 0.0)

    # 若所有权重都为 0，则退化成等权
    if w.sum() <= 0:
        w = np.ones_like(w)

    # 计算 Gaussian kernel
    z = (R_grid[:, None] - R_i[None, :]) / hR

    # 带权密度求和
    dens = (w[None, :] * (norm.pdf(z) / hR)).sum(axis=1)

    # 数值保护
    dens = np.maximum(dens, 0.0)

    # 归一化
    dens = dens / (dens.sum() * delta)

    return dens

# posterior_conf_good:
# 计算在 r=0, 样本量 N, 给定 prior 下，
# posterior 中 R >= 1 - p_bad_eff 的概率
def posterior_conf_good(N: int, prior_dens: np.ndarray, R_grid: np.ndarray, p_bad_eff: float) -> float:
    # 若 N 无效，则返回 nan
    if N is None or N <= 0:
        return np.nan

    # 网格步长
    delta = R_grid[1] - R_grid[0]

    # like = R^N，对应零失效时的似然
    like = np.power(R_grid, N)

    # 后验未归一化密度 = 先验 × 似然
    post = prior_dens * like

    # 归一化常数
    Z = post.sum() * delta

    # 若归一化常数非法，则返回 nan
    if Z <= 0:
        return np.nan

    # 可靠度阈值 R_thr = 1 - p_bad_eff
    R_thr = 1.0 - p_bad_eff

    # 找到所有 R >= R_thr 的网格点
    idx = (R_grid >= R_thr)

    # 数值积分得到 posterior probability
    return (post[idx].sum() * delta) / Z

# solve_Nstar_by_postconf:
# 从 N=1 到 N_max 搜索最小样本量 N*，
# 使得 posterior confidence >= conf_target
def solve_Nstar_by_postconf(prior_dens: np.ndarray, R_grid: np.ndarray, qhat: float,
                           p_bad: float, conf_target: float, N_max: int):
    # 经过 conformal 修正后的有效坏品率阈值
    p_bad_eff = clamp01(p_bad - qhat)

    # 从小到大搜索 N
    for N in range(1, N_max + 1):
        confN = posterior_conf_good(N, prior_dens, R_grid, p_bad_eff)

        # 若该 N 对应的置信度非法，则跳过
        if np.isnan(confN):
            continue

        # 找到第一个满足要求的 N 就返回
        if confN >= conf_target:
            return N

    # 如果直到 N_max 都没找到，则返回 None
    return None

# prior_pass_prob:
# 在先验分布下计算 E[R^N]
# 可理解为“先验通过概率”
def prior_pass_prob(N: int, prior_dens: np.ndarray, R_grid: np.ndarray) -> float:
    if N is None or N <= 0:
        return np.nan

    delta = R_grid[1] - R_grid[0]

    # 数值积分 E[R^N]
    out = (np.power(R_grid, N) * prior_dens).sum() * delta

    return float(np.clip(out, 0.0, 1.0))

# p_prior_implied:
# 已知先验通过概率 pass_prob，反推出等效失效率 p
# 满足 (1-p)^N = pass_prob
def p_prior_implied(N: int, pass_prob: float) -> float:
    if N is None or N <= 0 or np.isnan(pass_prob):
        return np.nan

    pp = float(np.clip(pass_prob, 0.0, 1.0))

    return 1.0 - (pp ** (1.0 / N))

# pick_writable_dir:
# 选择一个可写的输出目录
# 优先使用用户主目录下 output_A；若失败则退回临时目录
def pick_writable_dir() -> str:
    # 第一候选目录：用户主目录/output_A
    cand1 = os.path.join(os.path.expanduser("~"), "output_A")

    try:
        # 若目录不存在则创建
        os.makedirs(cand1, exist_ok=True)

        # 用写测试文件判断该目录是否可写
        testfile = os.path.join(cand1, ".write_test")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)

        return cand1

    except Exception:
        # 如果主目录不可写，则退回系统临时目录
        cand2 = os.path.join(tempfile.gettempdir(), "output_A")
        os.makedirs(cand2, exist_ok=True)
        return cand2


# =========================================================
# Simulation helpers
# =========================================================

# build_fixed_ppm_table:
# 根据 HIST_CLASS_DPPM 和 SEG_FACTOR 生成历史 ppm 表
# 输出列：类别, 段, X_ppm
def build_fixed_ppm_table() -> pd.DataFrame:
    # 对每个年份扩展 SEG_K 段，并乘以对应的 SEG_FACTOR
    base_seq = np.repeat(HIST_CLASS_DPPM, SEG_K) * np.tile(SEG_FACTOR, len(HIST_CLASS_NAMES))

    # 四舍五入成整数 ppm
    ppm = np.round(base_seq).astype(int)

    # 强制整体严格递减（与你原程序一致）
    for i in range(1, len(ppm)):
        if ppm[i] >= ppm[i - 1]:
            ppm[i] = ppm[i - 1] - 1

    # ppm 最低不能小于 1
    ppm = np.maximum(ppm, 1)

    # 生成年份列
    yrs = np.repeat(HIST_CLASS_NAMES, SEG_K)

    # 生成段编号列
    seg = np.tile(np.arange(1, SEG_K + 1), len(HIST_CLASS_NAMES))

    # 组成 DataFrame
    df = pd.DataFrame({"类别": yrs, "段": seg, "X_ppm": ppm})

    # 按类别和段排序
    df = df.sort_values(["类别", "段"]).reset_index(drop=True)

    return df

# simulate_hist_2020_2024:
# 按给定 ppm_table 模拟历史原始明细数据
def simulate_hist_2020_2024(seed: int, ppm_table: pd.DataFrame,
                           per_batch: int = 4000, n_per_seg_batch: int = 1) -> pd.DataFrame:
    # 创建随机数生成器
    rng = np.random.default_rng(seed)

    # 用于收集所有批次明细
    rows = []

    # 遍历每个历史类别
    for cls in HIST_CLASS_NAMES:
        # 遍历该类别下的每个段
        for s in range(1, SEG_K + 1):
            # 读取该类别该段对应的 ppm
            seg_ppm = int(ppm_table.loc[(ppm_table["类别"] == cls) & (ppm_table["段"] == s), "X_ppm"].iloc[0])

            # 转成失效率
            p_fail = seg_ppm / (10*1e6)

            # 每段生成若干个批次
            for b in range(1, n_per_seg_batch + 1):
                # 对每个样本做伯努利抽样，生成是否失效
                fail_vec = rng.binomial(1, p_fail, size=per_batch)

                # 存成一个明细 DataFrame
                rows.append(pd.DataFrame({
                    "批次编号": [f"{cls}_S{s}_L{b}"] * per_batch,
                    "类别": [cls] * per_batch,
                    "段": [s] * per_batch,
                    "X_ppm": [seg_ppm] * per_batch,
                    "是否失效": fail_vec.astype(int)
                }))

    # 合并所有批次明细
    return pd.concat(rows, ignore_index=True)

# simulate_2025_pool:
# 生成 2025 测试池的原始明细数据
def simulate_2025_pool(seed: int, n_batches: int, per_batch: int, p_true: float) -> pd.DataFrame:
    # 创建随机数生成器
    rng = np.random.default_rng(seed)

    # 总样本数 = 批次数 × 每批样本数
    total = n_batches * per_batch

    # 对所有样本做伯努利抽样，生成是否失效
    fail = rng.binomial(1, p_true, size=total).astype(int)

    # 构造批次编号
    batch_id = np.repeat(np.arange(1, n_batches + 1), per_batch)

    # 返回 DataFrame
    return pd.DataFrame({"批次编号": batch_id, "类别": "2025", "是否失效": fail})

# summarise_batches:
# 将原始历史明细按批次聚合成批次级 summary
def summarise_batches(df_hist: pd.DataFrame) -> pd.DataFrame:
    # 按批次编号、类别、段聚合
    g = df_hist.groupby(["批次编号", "类别", "段"], as_index=False).agg(
        n=("是否失效", "size"),     # 该批次样本数
        r=("是否失效", "sum"),      # 该批次失效数
        X_ppm=("X_ppm", "first")   # 该批次 ppm
    )

    # 批次经验可靠度 R_hat = 1 - r/n
    g["R_hat"] = np.clip(1.0 - g["r"] / g["n"], 0.0, 1.0)

    # 批次经验失效率 p_hat_batch = r/n
    g["p_hat_batch"] = np.clip(g["r"] / g["n"], 0.0, 1.0)

    # 本地真实失效率 proxy = X_ppm / 1e6
    g["p_true_local"] = g["X_ppm"] / 1e6

    return g


# =========================================================
# Convert simulation summary to company format
# =========================================================

# export_simulation_to_company_format:
# 将模拟得到的批次级 summary 转成公司要求的输入表格式
def export_simulation_to_company_format(batch_tbl_fixed: pd.DataFrame, output_path: str):
    """
    Convert simulated batch summary to company-style input format.
    One row = one historical batch summary.
    """

    # 构造公司格式表
    out = pd.DataFrame({
        "年份": batch_tbl_fixed["类别"].astype(str),
        "数量": batch_tbl_fixed["n"].astype(int),
        "失效数": batch_tbl_fixed["r"].astype(int),
        "全生命周期PPM": batch_tbl_fixed["X_ppm"].astype(float),
        "特征X1": np.nan,
        "特征X2": np.nan,
        "特征X3": np.nan,
        "特征X4": np.nan,
        "特征X5": np.nan,
        "特征X6": np.nan,
        "特征X7": np.nan,
        "特征X8": np.nan,
        "特征X9": np.nan,
    })

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 按文件扩展名决定保存为 csv 还是 excel
    if output_path.lower().endswith(".csv"):
        out.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        out.to_excel(output_path, index=False)

    # 打印保存位置
    print(f"[Saved] simulated company-format input: {output_path}")


# =========================================================
# Company-format loader
# =========================================================

# load_real_batch_table:
# 从公司格式表读取历史数据，并转成主程序内部使用的 batch_tbl_fixed 格式
def load_real_batch_table(file_path: str, sheet_name=0) -> pd.DataFrame:
    # 获取文件后缀的小写形式
    file_path_lower = str(file_path).lower()

    # 若是 csv，则直接读 csv
    if file_path_lower.endswith(".csv"):
        df = pd.read_csv(file_path)

    # 若是 Excel，则按指定 sheet 读取
    elif file_path_lower.endswith(".xlsx") or file_path_lower.endswith(".xls"):
        df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 否则报错
    else:
        raise ValueError("Unsupported input file type. Please use .csv, .xlsx, or .xls")

    # 去除列名首尾空格
    df.columns = [str(c).strip() for c in df.columns]

    # 公司格式最少需要这四列
    required_cols = ["年份", "数量", "失效数", "全生命周期PPM"]

    # 检查缺失列
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 自动识别所有特征列
    feature_cols = [c for c in df.columns if c.startswith("特征X")]

    # 只保留必须列和特征列
    out = df[required_cols + feature_cols].copy()

    # 年份转成字符串
    out["年份"] = out["年份"].astype(str).str.strip()

    # 数量转数值
    out["数量"] = pd.to_numeric(out["数量"], errors="coerce")

    # 失效数转数值
    out["失效数"] = pd.to_numeric(out["失效数"], errors="coerce")

    # 全生命周期PPM转数值
    out["全生命周期PPM"] = pd.to_numeric(out["全生命周期PPM"], errors="coerce")

    # 去掉关键字段缺失的行
    out = out.dropna(subset=["年份", "数量", "失效数", "全生命周期PPM"]).reset_index(drop=True)

    # 检查数量必须为正
    if (out["数量"] <= 0).any():
        raise ValueError("Found non-positive 数量 in input table")

    # 检查失效数不能为负
    if (out["失效数"] < 0).any():
        raise ValueError("Found negative 失效数 in input table")

    # 检查失效数不能大于数量
    if (out["失效数"] > out["数量"]).any():
        raise ValueError("Found 失效数 > 数量 in input table")

    # 检查全生命周期不能为负
    if (out["全生命周期PPM"] < 0).any():
        raise ValueError("Found negative 全生命周期PPM in input table")

    # 为每一行构造内部批次编号
    out["批次编号"] = [f"REAL_{i+1}" for i in range(len(out))]

    # 类别 = 年份
    out["类别"] = out["年份"]

    # 当前公司格式表没有段，统一设为1
    out["段"] = 1

    # 内部用 n 表示数量
    out["n"] = out["数量"].astype(int)

    # 内部用 r 表示失效数
    out["r"] = out["失效数"].astype(int)

    # 内部用 X_ppm 表示全生命周期PPM
    out["X_ppm"] = out["全生命周期PPM"].astype(float)

    # 经验可靠度 R_hat = 1 - r/n
    out["R_hat"] = np.clip(1.0 - out["r"] / out["n"], 0.0, 1.0)

    # 经验失效率 p_hat_batch = r/n
    out["p_hat_batch"] = np.clip(out["r"] / out["n"], 0.0, 1.0)

    # 用 X_ppm / 1e6 作为 local true rate proxy
    out["p_true_local"] = out["X_ppm"] / 1e6

    # 最终输出列顺序
    final_cols = [
        "批次编号", "类别", "段", "n", "r", "X_ppm",
        "R_hat", "p_hat_batch", "p_true_local"
    ] + feature_cols

    # 返回整理后的内部表
    return out[final_cols].copy()


# =========================================================
# Prior-related functions
# =========================================================

# prior_hist_given_x:
# 给定目标 x_ppm，构造历史部分的条件先验
def prior_hist_given_x(R_grid: np.ndarray, batch_tbl: pd.DataFrame,
                       x_ppm: int, hX: float, hR: float) -> np.ndarray:
    # 计算目标 ppm 与历史 ppm 的 Gaussian kernel 权重
    w = norm.pdf((x_ppm - batch_tbl["X_ppm"].to_numpy(dtype=float)) / hX)

    # 用权重对历史 R_hat 做 KDE
    return kde1d_density_weighted(R_grid, batch_tbl["R_hat"].to_numpy(dtype=float), w, hR)

# prior_new_from_proxy:
# 从 proxy 类别中构造 new prior
def prior_new_from_proxy(R_grid: np.ndarray, batch_tbl: pd.DataFrame, hR: float) -> np.ndarray:
    # 取出 proxy 类别与段对应的历史行
    tbl_new = batch_tbl[
        (batch_tbl["类别"].astype(str) == str(NEW_PROXY_CLASS)) &
        (batch_tbl["段"] == NEW_PROXY_SEG)
    ]

    # 若找不到，则退化成均匀密度
    if tbl_new.shape[0] == 0:
        delta = R_grid[1] - R_grid[0]
        dens = np.ones_like(R_grid, dtype=float)
        return dens / (dens.sum() * delta)

    # 否则对 proxy 的 R_hat 做 KDE
    return kde1d_density(R_grid, tbl_new["R_hat"].to_numpy(dtype=float), hR)

# mix_prior:
# 将历史 prior 和 new prior 按 alpha 混合
def mix_prior(prior_hist_x: np.ndarray, prior_new: np.ndarray, R_grid: np.ndarray, alpha: float) -> np.ndarray:
    # 网格步长
    delta = R_grid[1] - R_grid[0]

    # 混合
    prior = alpha * prior_hist_x + (1.0 - alpha) * prior_new

    # 数值保护
    prior = np.maximum(prior, 0.0)

    # 归一化
    prior = prior / (prior.sum() * delta)

    return prior

# estimate_qhat_weighted_split_fixed:
# 在固定 split 下，基于 calibration 集估计 qhat
def estimate_qhat_weighted_split_fixed(batch_tbl_fixed: pd.DataFrame,
                                       build_ids: np.ndarray, calib_ids: np.ndarray,
                                       alpha_now: float, R_grid: np.ndarray,
                                       hX: float, hR: float, conf_level: float) -> float:
    # build集
    build = batch_tbl_fixed.iloc[build_ids].copy()

    # calibration集
    calib = batch_tbl_fixed.iloc[calib_ids].copy()

    # 若 calibration 集为空，则返回 0
    if calib.shape[0] == 0:
        return 0.0

    # 基于 build 集构造 new prior
    prior_new_build = prior_new_from_proxy(R_grid, build, hR)

    # 网格步长
    delta = R_grid[1] - R_grid[0]

    # 存储每个 calibration 点的预测失效率
    phat = np.zeros(calib.shape[0], dtype=float)

    # calibration 点的“真实”局部失效率 proxy
    ptrue = calib["p_true_local"].to_numpy(dtype=float)

    # 对每个 calibration 点逐个构造条件先验
    for i in range(calib.shape[0]):
        # 当前 calibration 点的 ppm
        x_i = int(calib["X_ppm"].iloc[i])

        # 历史部分的条件先验
        prior_hist_xi = prior_hist_given_x(R_grid, build, x_i, hX=hX, hR=hR)

        # 混合 prior
        prior_xi = mix_prior(prior_hist_xi, prior_new_build, R_grid, alpha=alpha_now)

        # 计算 E(R | X=x_i)
        ER = (R_grid * prior_xi).sum() * delta

        # 预测失效率 = 1 - E(R)
        phat[i] = 1.0 - ER

    # 残差 = 真实失效率 proxy - 预测失效率
    resid = ptrue - phat

    # 取 conf_level 分位数作为 qhat
    qhat = float(np.quantile(resid, conf_level, method="linear"))

    return qhat


# =========================================================
# Step 1: generate simulated company-format input if needed
# =========================================================

# 若打开该开关，则先用模拟历史数据生成公司格式输入表
if GENERATE_SIM_COMPANY_INPUT:
    # 生成历史 ppm 表
    ppm_table_fixed_sim = build_fixed_ppm_table()

    # 固定历史模拟随机种子
    seed_hist_fixed_sim = SEED_BASE + 1

    # 生成历史明细数据
    df_hist_fixed_sim = simulate_hist_2020_2024(
        seed_hist_fixed_sim,
        ppm_table_fixed_sim,
        HIST_PER_SEG_BATCH,
        N_PER_SEG_BATCH
    )

    # 聚合成批次级 summary
    batch_tbl_fixed_sim = summarise_batches(df_hist_fixed_sim)

    # 导出成公司格式输入表
    export_simulation_to_company_format(batch_tbl_fixed_sim, REAL_INPUT_PATH)


# =========================================================
# Step 2: from now on, main program reads company-format input
# =========================================================

# 构造 R 的离散网格，用于数值积分
R_grid = np.arange(0.0, 1.0001, 0.001)

# 若使用公司格式输入，则读取公司表
if USE_REAL_INPUT:
    batch_tbl_fixed = load_real_batch_table(REAL_INPUT_PATH, REAL_INPUT_SHEET)

# 否则退回旧的模拟路径
else:
    ppm_table_fixed = build_fixed_ppm_table()
    seed_hist_fixed = SEED_BASE + 1
    df_hist_fixed = simulate_hist_2020_2024(
        seed_hist_fixed,
        ppm_table_fixed,
        HIST_PER_SEG_BATCH,
        N_PER_SEG_BATCH
    )
    batch_tbl_fixed = summarise_batches(df_hist_fixed)

# 找到 proxy 类别对应的行
proxy_rows = batch_tbl_fixed[
    (batch_tbl_fixed["类别"].astype(str) == str(NEW_PROXY_CLASS)) &
    (batch_tbl_fixed["段"] == NEW_PROXY_SEG)
]

# 若找不到 proxy，则报错
if proxy_rows.shape[0] == 0:
    raise RuntimeError(
        f"Cannot find proxy rows for NEW_PROXY_CLASS={NEW_PROXY_CLASS}, NEW_PROXY_SEG={NEW_PROXY_SEG}. "
        f"Please check your company-format input file."
    )

# 取 proxy 的 ppm 作为 x_new 基础值
x_new_ppm = int(proxy_rows["X_ppm"].iloc[0])

# 若采用 ×10 口径，则放大
if XNEW_TIMES_10:
    x_new_ppm = int(10 * x_new_ppm)

# 构造固定的历史部分先验（针对 x_new）
prior_hist_xnew_fixed = prior_hist_given_x(R_grid, batch_tbl_fixed, x_new_ppm, hX=HX_PPM, hR=hR)

# 构造固定的 proxy new prior
prior_new_fixed = prior_new_from_proxy(R_grid, batch_tbl_fixed, hR=hR)

# 固定随机 split
rng_split = np.random.default_rng(SPLIT_SEED_FIXED)

# 打乱所有批次索引
id_all = rng_split.permutation(batch_tbl_fixed.shape[0])

# build 集样本数
n_build = max(1, int(np.floor(BUILD_FRAC * batch_tbl_fixed.shape[0])))

# build 集索引
build_ids = id_all[:n_build]

# calibration 集索引
calib_ids = id_all[n_build:]

# 检查 calibration 集不为空
if calib_ids.size == 0:
    raise RuntimeError("Calibration split empty. Reduce BUILD_FRAC or increase data.")


# =========================================================
# Main loop
# =========================================================

# 存储每轮结果
rep_rows = []

# alpha 初始值
alpha_t = alpha_init

# 存储误差历史，用于 rolling mean
err_hist = []

# base ppm（如果开启了 ×10，则这里除回来）
x_new_ppm_base = int(x_new_ppm / 10) if XNEW_TIMES_10 else int(x_new_ppm)

# 基础先验失效率
p_prior_fixed_from_xnew = (x_new_ppm_base / 1e6)

# 选一个可写目录保存结果
safe_out_dir = pick_writable_dir()

# 打印输出目录
print(f"\nFiles will be saved to: {safe_out_dir}\n")

# 重复 N_REP 次
for t in range(1, N_REP + 1):
    # 第 t 轮的 2025 测试池随机种子
    seed_2025 = SEED_BASE + 999999 + t

    # 更新前的 alpha
    alpha_before = alpha_t

    # 构造当前 alpha 下的 x_new 混合先验
    prior_dens_xnew = mix_prior(prior_hist_xnew_fixed, prior_new_fixed, R_grid, alpha=alpha_before)

    # 用 fixed split 估计 qhat
    qhat_raw = estimate_qhat_weighted_split_fixed(
        batch_tbl_fixed, build_ids, calib_ids,
        alpha_now=alpha_before,
        R_grid=R_grid, hX=HX_PPM, hR=hR, conf_level=conf_level
    )

    # 若只允许更保守修正，则把 qhat 限制为非负
    qhat_use = max(qhat_raw, 0.0) if QHAT_ONLY_TIGHTEN else qhat_raw

    # 根据 posterior confidence 求 N*
    N_star = solve_Nstar_by_postconf(prior_dens_xnew, R_grid, qhat_use, P_BAD, CONF_TARGET, N_max)

    # 模拟 2025 测试池
    df_2025_pool = simulate_2025_pool(seed_2025, Y25_N_BATCHES_POOL, Y25_PER_BATCH_POOL, NEW_p_true_2025)

    # 测试池总样本数
    test_n = df_2025_pool.shape[0]

    # 默认不放回抽样
    replace_flag = False

    # 若 N* 大于测试池样本数，则看是否允许有放回抽样
    if N_star is not None and N_star > test_n:
        if TEST_SAMPLE_REPLACE_IF_NEEDED:
            replace_flag = True
        else:
            N_star = None

    # 初始化观测统计量
    F_t = None
    p_obs_t = np.nan
    passed = None

    # 若 N* 有效，则抽样并计算观测失效率
    if N_star is not None and N_star > 0:
        rng = np.random.default_rng(seed_2025 + 777)
        idx = rng.choice(test_n, size=N_star, replace=replace_flag)
        F_t = int(df_2025_pool["是否失效"].to_numpy()[idx].sum())
        p_obs_t = F_t / N_star
        passed = int(pass_rule_test(F_t))

    # 计算 implied prior pass probability
    pass_prob = prior_pass_prob(N_star, prior_dens_xnew, R_grid) if N_star is not None else np.nan

    # 反推 implied p
    p_prior_imp = p_prior_implied(N_star, pass_prob) if N_star is not None else np.nan

    # 用户当前口径：p_prior_t = base × 10（all-life）
    p_prior_t = p_prior_fixed_from_xnew * 10.0

    # 单次校准误差
    err_t = (p_obs_t - p_prior_t) if not np.isnan(p_obs_t) else np.nan

    # 将有效误差加入历史
    if not np.isnan(err_t):
        err_hist.append(err_t)

    # 初始化平滑误差
    err_smooth = np.nan

    # 若已有误差历史，则做 rolling mean
    if len(err_hist) > 0:
        D = min(WIN, len(err_hist))
        err_smooth = float(np.mean(err_hist[-D:]))

        # 用平滑误差更新 alpha
        delta_alpha = clamp_step(STEP_SIZE * (err_smooth / ERR_SCALE))
        alpha_t = clamp_alpha(alpha_t - delta_alpha)

    # 更新后的 alpha
    alpha_after = alpha_t

    # 保存本轮结果
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

    # 每 10 轮打印一次日志
    if t % 10 == 0:
        print(f"t={t} alpha={alpha_before:.3f}->{alpha_after:.3f} "
              f"N*={N_star} pass={passed} "
              f"p_obs={p_obs_t if not np.isnan(p_obs_t) else 'NA'} "
              f"p_prior_xnew={p_prior_t:.6g} "
              f"err={err_t if not np.isnan(err_t) else 'NA'} "
              f"WinMean={err_smooth if not np.isnan(err_smooth) else 'NA'}")

# 将重复实验结果汇总成 DataFrame
rep_df = pd.DataFrame(rep_rows)


# =========================================================
# 计算置信度表
# =========================================================

# 1) 可选ppm
PPM_USED_GRID = [
    200, 800, 1400, 2000,
    4000, 6000, 8000, 10000, 12000, 14000,
    16000, 18000, 20000, 22000, 24000, 26000,
    28000, 30000
]


t_anchor = 1

N_fixed = int(rep_df.loc[rep_df["rep_id"] == t_anchor, "N_star"].iloc[0])

alpha_anchor = float(rep_df.loc[rep_df["rep_id"] == t_anchor, "alpha_before"].iloc[0])
qhat_anchor  = float(rep_df.loc[rep_df["rep_id"] == t_anchor, "qhat_used"].iloc[0])


def build_prior_dens_for_usedppm(ppm_used: int) -> np.ndarray:
    x_used = int(round(ppm_used))

    prior_hist_x = prior_hist_given_x(
        R_grid,
        batch_tbl_fixed,
        x_ppm=x_used,
        hX=HX_PPM,
        hR=hR
    )

    prior_dens_x = mix_prior(
        prior_hist_x,
        prior_new_fixed,
        R_grid,
        alpha=alpha_anchor
    )

    return prior_dens_x


def calc_fixedN_confidence_row(ppm_used: int) -> dict:

    p_bad_row = ppm_used / 1e6

    p_bad_eff_row = clamp01(p_bad_row - qhat_anchor)
    
    prior_dens_x = build_prior_dens_for_usedppm(ppm_used)

    conf_row = posterior_conf_good(
        N=N_fixed,
        prior_dens=prior_dens_x,
        R_grid=R_grid,
        p_bad_eff=p_bad_eff_row
    )

    meets_target = (conf_row >= CONF_TARGET) if pd.notna(conf_row) else False

    return {
        "X_new*10 ppm": ppm_used,
        "抽样样本量 N*": N_fixed,     
        "失效个数 r": 0,             
        "confidence": conf_row,     
        "CONF_TARGET": CONF_TARGET,  
        "是否达到CONF_TARGET": meets_target,
        "qhat_anchor": qhat_anchor,
        "alpha_anchor": alpha_anchor,
        "p_bad_row": p_bad_row,
        "p_bad_eff_row": p_bad_eff_row
    }


# 5) 生成最终表
fixedN_conf_tbl = pd.DataFrame(
    [calc_fixedN_confidence_row(ppm) for ppm in PPM_USED_GRID]
)

print("\n===== Fixed-N confidence table (replace LTPD by X_new*10 ppm) =====")
print(fixedN_conf_tbl)


fixedN_conf_tbl_fmt = fixedN_conf_tbl.copy()
fixedN_conf_tbl_fmt["confidence"] = fixedN_conf_tbl_fmt["confidence"].map(
    lambda x: f"{x:.7f}" if pd.notna(x) else "NA"
)
fixedN_conf_tbl_fmt["CONF_TARGET"] = fixedN_conf_tbl_fmt["CONF_TARGET"].map(
    lambda x: f"{x:.2f}"
)

print("\n===== Formatted Fixed-N confidence table =====")
print(fixedN_conf_tbl_fmt)


# 7) save CSV
fn_fixedN_conf_tbl = os.path.join(
    safe_out_dir,
    f"fixedN_confidence_table_from_xnew10ppm_t{t_anchor}_2025ppm{NEW_2025_DPPM}_NREP{N_REP}.csv"
)
fixedN_conf_tbl.to_csv(fn_fixedN_conf_tbl, index=False, encoding="utf-8-sig")

print("\n[Saved] Fixed-N confidence table CSV:", fn_fixedN_conf_tbl)

# 生成 summary 表
sum_df = pd.DataFrame([{
    "N_rep": rep_df.shape[0],
    "pass_rate": float(rep_df["pass"].dropna().mean()) if rep_df["pass"].notna().any() else np.nan,
    "alpha_start": float(rep_df["alpha_before"].iloc[0]),
    "alpha_end": float(rep_df["alpha_after"].iloc[-1]),
    "Nstar_mean": float(rep_df["N_star"].dropna().mean()) if rep_df["N_star"].notna().any() else np.nan,
    "Nstar_min": int(rep_df["N_star"].dropna().min()) if rep_df["N_star"].notna().any() else None,
    "Nstar_max": int(rep_df["N_star"].dropna().max()) if rep_df["N_star"].notna().any() else None,
}])

# 打印 summary
print("\n===== SUMMARY =====")
print(sum_df)

# 保存 rep_df
fn_rep = os.path.join(safe_out_dir, f"A_direct_scheme1_2025ppm_20250101{NEW_2025_DPPM}{N_REP}.csv")

# 保存 summary
fn_sum = os.path.join(safe_out_dir, f"A_direct_scheme1_2025ppm_20250101{NEW_2025_DPPM}{N_REP}_summary.csv")

rep_df.to_csv(fn_rep, index=False)
sum_df.to_csv(fn_sum, index=False)

# 打印输出文件路径
print("\nSaved:")
print(" input company-format file:", REAL_INPUT_PATH)
print(" rep_df:", fn_rep)
print(" sum_df:", fn_sum)
