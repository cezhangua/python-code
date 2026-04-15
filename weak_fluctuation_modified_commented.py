import os  # 导入 os，用于处理路径、目录创建和文件路径拼接
import tempfile  # 导入 tempfile，用于在主目录不可写时回退到临时目录
import numpy as np  # 导入 numpy，用于数组、数值计算和随机数生成
import pandas as pd  # 导入 pandas，用于表格数据整理与导出
from scipy.stats import norm  # 导入标准正态分布函数，用于 Gaussian 核加权


# =========================================================
# Global params
# =========================================================

LTPD_TOTAL_DEFAULT = 0.01          # 默认 LTPD 占位值，主程序中实际使用的是后面 LTPD_PPM_GRID 转换后的数值
conf_level = 0.90                  # Conformal 分位数水平，当前取 90%
hR = 0.02                          # 可靠率方向带宽，用于对历史 R_hat 做 KDE 平滑
N_max = 6000                       # 反解样本量时允许搜索的最大 N 上限

r_allow = 0                        # 零失效通过规则：允许的最大失效数为 0
def pass_rule_test(x_fail: int) -> bool:  # 定义通过判据函数，输入为抽样失效个数
    return x_fail <= r_allow              # 若失效数不超过允许值，则返回通过

N_REP = 100                        # 每个 (2025 ppm, LTPD) 组合重复仿真的次数
SEED_BASE = 20250101               # 统一随机种子基准，用于保证结果可复现

# 弱波动场景：仍沿用你之前较平滑下降的设定
HIST_CLASS_NAMES = ["2020", "2021", "2022", "2023", "2024"]  # 历史年份类别列表
HIST_CLASS_DPPM = np.array([10000, 5000, 2000, 1000, 200], dtype=float)  # 各年份基础 ppm，整体平滑下降

SEG_K = 5                          # 每个年份再划分为 5 个有序分段
SEG_FACTOR = np.array([1.30, 1.15, 1.00, 0.85, 0.70], dtype=float)  # 年内 5 段的相对波动因子
SEG_FACTOR = SEG_FACTOR / SEG_FACTOR.mean()  # 归一化分段因子，使平均因子为 1

NEW_PROXY_CLASS = "2024"           # 当前目标批次的代理参考年份
NEW_PROXY_SEG = 1                  # 当前目标批次的代理参考分段

Y25_N_BATCHES_POOL = 10            # 2025 测试池中的批次数
Y25_PER_BATCH_POOL = 4000          # 2025 每个批次的样本数
TEST_SAMPLE_REPLACE_IF_NEEDED = True  # 若 N* 大于测试池总量，允许有放回抽样

HX_PPM = 500                       # ppm 方向带宽，用于目标 ppm 与历史 ppm 的相似性加权

# 时间变量：年份 + 段 -> 顺序时间索引
BASE_YEAR = 2020                   # 顺序时间索引的起始年份
HT_TIME = 5                        # 时间方向带宽，用于控制时间权重衰减

alpha_init = 0.999                 # 在线更新时历史权重 alpha 的初始值，接近完全继承历史
ALPHA_MIN = 0.001                  # alpha 的最小允许值
ALPHA_MAX = 0.999                  # alpha 的最大允许值
STEP_SIZE = 1.0                    # 在线更新步长系数
DELTA_MAX = 0.01                   # 每轮 alpha 的最大更新幅度
WIN = 10                           # 在线更新时的误差滑动窗口长度

BUILD_FRAC = 0.50                  # 历史批次中 build set 的占比
SPLIT_SEED_FIXED = SEED_BASE + 55555  # 固定的 build/calibration 划分随机种子

HIST_PER_SEG_BATCH = 4000          # 每个历史分段批次的样本量
N_PER_SEG_BATCH = 1                # 每个历史分段生成的批次数

CONF_TARGET = 0.90                 # 反解样本量时要求达到的后验置信门槛

# 这里特别注意：这次不再对 qhat 截断
QHAT_ONLY_TIGHTEN = False          # 本程序保留原始 qhat，不再做 max(qhat, 0)

XNEW_TIMES_10 = True               # 是否把代理输入 ppm 放大 10 倍再用于先验构造

# 输出路径
OUTPUT_BASENAME = "weak_fluctuation_ltpd_grid"  # 输出文件名前缀


# =========================================================
# Company-format input switches
# =========================================================

GENERATE_SIM_COMPANY_INPUT = True  # 是否先把当前模拟历史数据导出成公司要求的 Excel 输入格式
USE_REAL_INPUT = True  # 后续主程序是否优先从公司格式 Excel / CSV 中读取历史输入
REAL_INPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "company_input_from_simulation.xlsx")  # 公司格式输入文件路径（默认放在脚本同目录）
REAL_INPUT_SHEET = 0  # 若为 Excel，默认读取第 1 个 sheet


# =========================================================
# 用户可调网格
# =========================================================

# 2025 最新产品 ppm 网格（你可以继续改）
PPM_2025_GRID_GOOD = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1400, 2000, 3000]  # 视为好批次的 ppm 网格
PPM_2025_GRID_BAD  = [4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000]        # 视为坏批次的 ppm 网格

PPM_2025_GRID_ALL = PPM_2025_GRID_GOOD + PPM_2025_GRID_BAD  # 合并为完整的 2025 ppm 网格

# 客户 LTPD（ppm）网格
# LTPD 网格，已放到 10000
LTPD_PPM_GRID = [200, 400, 600, 800, 1000, 1400, 2000, 3000, 4000, 6000, 8000, 10000]  # 客户 LTPD 网格

# 这个值用于单独做“好批次/坏批次表”
# 例如想复现类似你截图的某一张表，可以指定一个客户 LTPD
SELECTED_LTPD_PPM_FOR_SUMMARY = 10000  # 用于公司风格摘要表的固定 LTPD


# =========================================================
# Utils
# =========================================================

def clamp01(x: float) -> float:  # 把输入裁剪到 [0,1] 区间
    return max(0.0, min(1.0, float(x)))  # 先转成浮点数，再进行上下界截断

def clamp_alpha(a: float) -> float:  # 把 alpha 裁剪到允许区间
    return max(ALPHA_MIN, min(ALPHA_MAX, float(a)))  # 防止 alpha 过小或过大

def clamp_step(d: float) -> float:  # 把单轮 alpha 更新量裁剪到 [-DELTA_MAX, DELTA_MAX]
    return max(-DELTA_MAX, min(DELTA_MAX, float(d)))  # 限制每轮更新幅度

def make_time_index(year_str, seg, base_year=BASE_YEAR, seg_k=SEG_K):  # 构造顺序时间索引 tau(c,s)
    year_num = int(year_str)  # 把年份字符串转成整数
    return (year_num - base_year) * seg_k + (int(seg) - 1)  # 计算该 年份×分段 在总时间链条中的位置

def kde1d_density(R_grid: np.ndarray, R_i: np.ndarray, hR: float) -> np.ndarray:  # 不加权的一维 KDE 密度估计
    delta = R_grid[1] - R_grid[0]  # 计算可靠率网格步长，用于数值积分归一化

    if R_i.size == 0:  # 如果没有历史可靠率点
        dens = np.ones_like(R_grid, dtype=float)  # 用均匀密度占位
        return dens / (dens.sum() * delta)  # 归一化后返回

    z = (R_grid[:, None] - R_i[None, :]) / hR  # 计算每个网格点到各历史点的标准化距离
    dens = norm.pdf(z) / hR  # 计算 Gaussian 核密度值
    dens = dens.sum(axis=1)  # 对所有历史点求和，得到每个网格点的总密度
    dens = np.maximum(dens, 0.0)  # 防止数值误差导致负值
    dens = dens / (dens.sum() * delta)  # 对密度做离散归一化
    return dens  # 返回归一化后的密度

def kde1d_density_weighted(R_grid: np.ndarray, R_i: np.ndarray, w_i: np.ndarray, hR: float) -> np.ndarray:  # 加权的一维 KDE 密度估计
    delta = R_grid[1] - R_grid[0]  # 计算可靠率网格步长

    if R_i.size == 0:  # 如果没有历史可靠率点
        dens = np.ones_like(R_grid, dtype=float)  # 用均匀密度占位
        return dens / (dens.sum() * delta)  # 归一化后返回

    w = np.maximum(w_i.astype(float), 0.0)  # 把权重转成非负浮点数
    if w.sum() <= 0:  # 如果总权重异常
        w = np.ones_like(w)  # 回退为等权重

    z = (R_grid[:, None] - R_i[None, :]) / hR  # 计算网格点到各历史可靠率点的标准化距离
    dens = (w[None, :] * (norm.pdf(z) / hR)).sum(axis=1)  # 乘以权重后对各历史点求和
    dens = np.maximum(dens, 0.0)  # 防止数值误差
    dens = dens / (dens.sum() * delta)  # 归一化得到有效密度
    return dens  # 返回加权 KDE 密度

def posterior_conf_good(N: int, prior_dens: np.ndarray, R_grid: np.ndarray, p_bad_eff: float) -> float:  # 计算给定 N 下的后验“好批次概率”
    if N is None or N <= 0:  # 若样本量无效
        return np.nan  # 返回缺失值

    delta = R_grid[1] - R_grid[0]  # 计算可靠率网格步长
    like = np.power(R_grid, N)  # 在零失效条件下，似然项为 R^N
    post = prior_dens * like  # 未归一化后验 = 先验 × 似然
    Z = post.sum() * delta  # 计算归一化常数

    if Z <= 0:  # 若后验归一化常数异常
        return np.nan  # 返回缺失值

    R_thr = 1.0 - p_bad_eff  # 把有效坏品率阈值转成可靠率阈值
    idx = (R_grid >= R_thr)  # 找到满足可靠率不低于阈值的网格位置
    return (post[idx].sum() * delta) / Z  # 返回该区域下的后验概率质量

def solve_Nstar_by_postconf(prior_dens: np.ndarray, R_grid: np.ndarray, qhat: float,
                            p_bad: float, conf_target: float, N_max: int):  # 搜索最小样本量 N*
    p_bad_eff = clamp01(p_bad - qhat)  # 构造经 Conformal 修正后的有效坏品率门槛

    for N in range(1, N_max + 1):  # 从 1 开始逐个尝试样本量
        confN = posterior_conf_good(N, prior_dens, R_grid, p_bad_eff)  # 计算该 N 下的后验好批次概率
        if np.isnan(confN):  # 若该 N 计算失败
            continue  # 跳过当前 N
        if confN >= conf_target:  # 如果已经达到目标置信门槛
            return N  # 返回最小满足条件的 N
    return None  # 若找不到，则返回 None

def prior_pass_prob(N: int, prior_dens: np.ndarray, R_grid: np.ndarray) -> float:  # 计算先验下零失效通过概率 E[R^N]
    if N is None or N <= 0:  # 若样本量无效
        return np.nan  # 返回缺失值

    delta = R_grid[1] - R_grid[0]  # 计算可靠率网格步长
    out = (np.power(R_grid, N) * prior_dens).sum() * delta  # 数值积分计算 E[R^N]
    return float(np.clip(out, 0.0, 1.0))  # 截断到 [0,1] 并返回

def p_prior_implied(N: int, pass_prob: float) -> float:  # 根据先验通过概率反推出等效失效率
    if N is None or N <= 0 or np.isnan(pass_prob):  # 若输入无效
        return np.nan  # 返回缺失值

    pp = float(np.clip(pass_prob, 0.0, 1.0))  # 把通过概率裁剪到 [0,1]
    return 1.0 - (pp ** (1.0 / N))  # 根据 (1-p)^N = pass_prob 反解等效 p



def posterior_density_zero_fail(N: int, prior_dens: np.ndarray, R_grid: np.ndarray) -> np.ndarray:  # 在零失效条件下构造后验密度
    delta = R_grid[1] - R_grid[0]  # 计算可靠率网格步长

    if N is None or N <= 0:  # 若样本量无效
        dens = np.ones_like(R_grid, dtype=float)  # 用均匀密度占位
        return dens / (dens.sum() * delta)  # 返回归一化后的占位密度

    like = np.power(R_grid, N)  # 零失效条件下的似然项为 R^N
    post = np.maximum(prior_dens * like, 0.0)  # 未归一化后验 = 先验 × 似然
    Z = post.sum() * delta  # 计算归一化常数

    if Z <= 0:  # 若归一化常数异常
        dens = np.ones_like(R_grid, dtype=float)  # 用均匀密度回退
        return dens / (dens.sum() * delta)  # 返回归一化后的占位密度

    return post / Z  # 返回零失效条件下的后验密度


def posterior_mean_R_from_density(post_dens: np.ndarray, R_grid: np.ndarray) -> float:  # 根据后验密度计算后验平均可靠率
    delta = R_grid[1] - R_grid[0]  # 计算可靠率网格步长
    return float(np.clip((R_grid * post_dens).sum() * delta, 0.0, 1.0))  # 返回后验平均可靠率


def sample_R_from_density(seed: int, post_dens: np.ndarray, R_grid: np.ndarray) -> float:  # 从后验可靠率分布中抽取一个 R 值
    rng = np.random.default_rng(seed)  # 初始化随机数生成器
    delta = R_grid[1] - R_grid[0]  # 计算可靠率网格步长
    prob = np.maximum(post_dens.astype(float), 0.0) * delta  # 把密度离散化为概率质量

    if prob.sum() <= 0:  # 若概率质量异常
        prob = np.ones_like(prob) / prob.size  # 回退为均匀分布
    else:  # 若概率质量正常
        prob = prob / prob.sum()  # 归一化为离散分布

    idx = int(rng.choice(prob.size, p=prob))  # 按离散后验分布抽取一个网格点索引
    return float(np.clip(R_grid[idx], 0.0, 1.0))  # 返回抽到的可靠率


def simulate_2025_pool_from_posterior(seed: int, n_batches: int, per_batch: int, post_dens: np.ndarray, R_grid: np.ndarray):  # 用后验分布生成新的 2025 测试池
    R_draw = sample_R_from_density(seed, post_dens, R_grid)  # 从后验分布中抽取一个可靠率
    p_draw = float(np.clip(1.0 - R_draw, 0.0, 1.0))  # 转成对应的失效率
    df_pool = simulate_2025_pool(seed=seed + 123, n_batches=n_batches, per_batch=per_batch, p_true=p_draw)  # 按该失效率生成新的测试池
    return df_pool, R_draw, p_draw  # 返回后验生成的测试池及其对应的 R / p

def pick_writable_dir() -> str:  # 选择一个可写的输出目录
    cand1 = os.path.join(os.path.expanduser("~"), "output_A")  # 优先尝试用户主目录下的 output_A

    try:  # 尝试创建并测试首选目录
        os.makedirs(cand1, exist_ok=True)  # 若目录不存在则创建
        testfile = os.path.join(cand1, ".write_test")  # 构造测试写入文件路径
        with open(testfile, "w") as f:  # 尝试写文件
            f.write("ok")  # 写入简单内容
        os.remove(testfile)  # 删除测试文件
        return cand1  # 若成功，则返回首选目录

    except Exception:  # 如果首选目录不可写
        cand2 = os.path.join(tempfile.gettempdir(), "output_A")  # 回退到系统临时目录
        os.makedirs(cand2, exist_ok=True)  # 确保临时目录存在
        return cand2  # 返回临时目录


# =========================================================
# Simulation helpers
# =========================================================

def build_fixed_ppm_table() -> pd.DataFrame:  # 构造弱波动场景的固定历史 ppm 表
    base_seq = np.repeat(HIST_CLASS_DPPM, SEG_K) * np.tile(SEG_FACTOR, len(HIST_CLASS_NAMES))  # 生成年份基准 ppm × 年内因子的序列
    ppm = np.round(base_seq).astype(int)  # 四舍五入并转成整数 ppm

    # 弱波动场景：保持整体严格递减（沿用你之前弱波动逻辑）
    for i in range(1, len(ppm)):  # 从第二个历史单元开始检查
        if ppm[i] >= ppm[i - 1]:  # 如果当前 ppm 不小于前一个 ppm
            ppm[i] = ppm[i - 1] - 1  # 强制减 1，确保整体严格递减

    ppm = np.maximum(ppm, 1)  # 防止 ppm 变成 0 或负数

    yrs = np.repeat(HIST_CLASS_NAMES, SEG_K)  # 生成每个历史单元对应的年份列
    seg = np.tile(np.arange(1, SEG_K + 1), len(HIST_CLASS_NAMES))  # 生成每个历史单元对应的分段列

    df = pd.DataFrame({"类别": yrs, "段": seg, "X_ppm": ppm})  # 组装成 DataFrame
    df = df.sort_values(["类别", "段"]).reset_index(drop=True)  # 按年份和分段排序并重置索引
    return df  # 返回固定历史 ppm 表

def simulate_hist_2020_2024(seed: int, ppm_table: pd.DataFrame,
                            per_batch: int = 4000, n_per_seg_batch: int = 1) -> pd.DataFrame:  # 根据历史 ppm 表生成历史样本级数据
    rng = np.random.default_rng(seed)  # 初始化随机数生成器
    rows = []  # 用于收集每个历史批次的 DataFrame

    for cls in HIST_CLASS_NAMES:  # 遍历每个历史年份
        for s in range(1, SEG_K + 1):  # 遍历每个年份内的分段
            seg_ppm = int(  # 取出该 年份×分段 的 ppm
                ppm_table.loc[
                    (ppm_table["类别"] == cls) & (ppm_table["段"] == s),
                    "X_ppm"
                ].iloc[0]
            )

            # 保持你原来的模拟口径，不额外改
            p_fail = seg_ppm / (10 * 1e6)  # 把 ppm 转成模拟所用的单颗失效率口径

            for b in range(1, n_per_seg_batch + 1):  # 遍历该分段下要生成的批次数
                fail_vec = rng.binomial(1, p_fail, size=per_batch)  # 为该批次生成每颗芯片是否失效
                rows.append(pd.DataFrame({  # 把该批次结果整理成 DataFrame 并加入列表
                    "批次编号": [f"{cls}_S{s}_L{b}"] * per_batch,
                    "类别": [cls] * per_batch,
                    "段": [s] * per_batch,
                    "X_ppm": [seg_ppm] * per_batch,
                    "是否失效": fail_vec.astype(int)
                }))

    return pd.concat(rows, ignore_index=True)  # 合并所有批次并返回

def simulate_2025_pool(seed: int, n_batches: int, per_batch: int, p_true: float) -> pd.DataFrame:  # 生成 2025 当前目标批次测试池
    rng = np.random.default_rng(seed)  # 初始化随机数生成器
    total = n_batches * per_batch  # 计算总样本量
    fail = rng.binomial(1, p_true, size=total).astype(int)  # 按真实失效率生成失效指示
    batch_id = np.repeat(np.arange(1, n_batches + 1), per_batch)  # 构造批次编号

    return pd.DataFrame({"批次编号": batch_id, "类别": "2025", "是否失效": fail})  # 返回 2025 测试池

def summarise_batches(df_hist: pd.DataFrame) -> pd.DataFrame:  # 把历史样本级数据汇总成批次级摘要
    g = df_hist.groupby(["批次编号", "类别", "段"], as_index=False).agg(  # 按 批次编号×类别×分段 分组
        n=("是否失效", "size"),  # 该批次样本量
        r=("是否失效", "sum"),  # 该批次失效数
        X_ppm=("X_ppm", "first")  # 该批次对应的 ppm
    )

    g["R_hat"] = np.clip(1.0 - g["r"] / g["n"], 0.0, 1.0)  # 经验可靠率 R_hat
    g["p_hat_batch"] = np.clip(g["r"] / g["n"], 0.0, 1.0)  # 经验失效率
    g["p_true_local"] = g["X_ppm"] / 1e6  # 把 ppm 转成局部真失效率代理值
    return g  # 返回批次摘要表


# =========================================================
# Company-format conversion helpers
# =========================================================

def export_simulation_to_company_format(batch_tbl_fixed: pd.DataFrame, output_path: str):  # 把当前模拟历史数据导出成公司要求的输入格式
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
    })  # 构造公司输入格式表

    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保输出目录存在

    if output_path.lower().endswith(".csv"):  # 若目标为 CSV
        out.to_csv(output_path, index=False, encoding="utf-8-sig")  # 导出 CSV
    else:  # 其余情况按 Excel 导出
        out.to_excel(output_path, index=False)  # 导出 Excel

    print(f"[Saved] simulated company-format input: {output_path}")  # 打印导出路径


def load_real_batch_table(file_path: str, sheet_name=0) -> pd.DataFrame:  # 从公司格式 Excel / CSV 中读取历史输入
    file_path_lower = str(file_path).lower()  # 统一成小写路径后缀

    if file_path_lower.endswith(".csv"):  # 若输入为 CSV
        df = pd.read_csv(file_path)  # 读取 CSV
    elif file_path_lower.endswith(".xlsx") or file_path_lower.endswith(".xls"):  # 若输入为 Excel
        df = pd.read_excel(file_path, sheet_name=sheet_name)  # 读取 Excel
    else:  # 若文件类型不支持
        raise ValueError("Unsupported input file type. Please use .csv, .xlsx, or .xls")  # 抛出错误

    df.columns = [str(c).strip() for c in df.columns]  # 去掉列名空格

    required_cols = ["年份", "数量", "失效数", "全生命周期PPM"]  # 公司输入格式的必需列
    missing = [c for c in required_cols if c not in df.columns]  # 检查缺失列
    if missing:  # 若缺列
        raise ValueError(f"Missing required columns: {missing}")  # 抛出错误

    feature_cols = [c for c in df.columns if c.startswith("特征X")]  # 识别附加特征列
    out = df[required_cols + feature_cols].copy()  # 保留核心列和特征列

    out["年份"] = out["年份"].astype(str).str.strip()  # 统一年份为字符串
    out["数量"] = pd.to_numeric(out["数量"], errors="coerce")  # 转数量为数值
    out["失效数"] = pd.to_numeric(out["失效数"], errors="coerce")  # 转失效数为数值
    out["全生命周期PPM"] = pd.to_numeric(out["全生命周期PPM"], errors="coerce")  # 转 ppm 为数值

    out = out.dropna(subset=["年份", "数量", "失效数", "全生命周期PPM"]).reset_index(drop=True)  # 删除关键字段缺失行

    if (out["数量"] <= 0).any():  # 检查数量是否为正
        raise ValueError("Found non-positive 数量 in input table")  # 抛出错误
    if (out["失效数"] < 0).any():  # 检查失效数是否为负
        raise ValueError("Found negative 失效数 in input table")  # 抛出错误
    if (out["失效数"] > out["数量"]).any():  # 检查失效数是否超过数量
        raise ValueError("Found 失效数 > 数量 in input table")  # 抛出错误
    if (out["全生命周期PPM"] < 0).any():  # 检查 ppm 是否为负
        raise ValueError("Found negative 全生命周期PPM in input table")  # 抛出错误

    out["批次编号"] = [f"REAL_{i+1}" for i in range(len(out))]  # 生成人工批次编号
    out["类别"] = out["年份"]  # 类别列直接使用年份
    out["段"] = out.groupby("年份").cumcount() + 1  # 按年份内出现顺序自动恢复分段编号 1,2,...

    out["n"] = out["数量"].astype(int)  # 映射成程序内部数量列 n
    out["r"] = out["失效数"].astype(int)  # 映射成程序内部失效数列 r
    out["X_ppm"] = out["全生命周期PPM"].astype(float)  # 映射成程序内部 ppm 列

    out["R_hat"] = np.clip(1.0 - out["r"] / out["n"], 0.0, 1.0)  # 计算经验可靠率
    out["p_hat_batch"] = np.clip(out["r"] / out["n"], 0.0, 1.0)  # 计算经验失效率
    out["p_true_local"] = out["X_ppm"] / 1e6  # 将 ppm 换算成局部失效率代理值

    final_cols = [
        "批次编号", "类别", "段", "n", "r", "X_ppm",
        "R_hat", "p_hat_batch", "p_true_local"
    ] + feature_cols  # 最终返回列

    return out[final_cols].copy()  # 返回程序内部所需格式


# =========================================================
# Prior-related functions
# =========================================================

def prior_hist_given_x(R_grid: np.ndarray, batch_tbl: pd.DataFrame,
                       x_ppm: int, target_time_index: float,
                       hX: float, hT: float, hR: float) -> np.ndarray:  # 构造给定目标输入下的历史条件先验
    w_ppm = norm.pdf((x_ppm - batch_tbl["X_ppm"].to_numpy(dtype=float)) / hX)  # 计算 ppm 相似性权重

    hist_time_index = np.array([  # 计算每个历史批次的顺序时间索引
        make_time_index(y, s, base_year=BASE_YEAR, seg_k=SEG_K)
        for y, s in zip(batch_tbl["类别"].astype(str), batch_tbl["段"])
    ], dtype=float)

    w_time = norm.pdf((target_time_index - hist_time_index) / hT)  # 计算时间相似性权重

    w = w_ppm * w_time  # 联合权重 = ppm 权重 × 时间权重

    return kde1d_density_weighted(  # 用联合权重对历史 R_hat 做加权 KDE
        R_grid,
        batch_tbl["R_hat"].to_numpy(dtype=float),
        w,
        hR
    )

def prior_new_from_proxy(R_grid: np.ndarray, batch_tbl: pd.DataFrame, hR: float) -> np.ndarray:  # 构造最近参考批次先验
    tbl_new = batch_tbl[  # 选出代理历史批次
        (batch_tbl["类别"].astype(str) == str(NEW_PROXY_CLASS)) &
        (batch_tbl["段"] == NEW_PROXY_SEG)
    ]

    if tbl_new.shape[0] == 0:  # 若代理批次不存在
        delta = R_grid[1] - R_grid[0]  # 计算网格步长
        dens = np.ones_like(R_grid, dtype=float)  # 用均匀密度占位
        return dens / (dens.sum() * delta)  # 归一化后返回

    return kde1d_density(R_grid, tbl_new["R_hat"].to_numpy(dtype=float), hR)  # 对代理批次的 R_hat 做 KDE

def mix_prior(prior_hist_x: np.ndarray, prior_new: np.ndarray, R_grid: np.ndarray, alpha: float) -> np.ndarray:  # 按 alpha 混合历史先验和最近先验
    delta = R_grid[1] - R_grid[0]  # 计算网格步长
    prior = alpha * prior_hist_x + (1.0 - alpha) * prior_new  # 线性混合两部分先验
    prior = np.maximum(prior, 0.0)  # 防止数值误差
    prior = prior / (prior.sum() * delta)  # 归一化为有效密度
    return prior  # 返回混合先验密度

def estimate_qhat_weighted_split_fixed(batch_tbl_fixed: pd.DataFrame,
                                       build_ids: np.ndarray, calib_ids: np.ndarray,
                                       alpha_now: float, R_grid: np.ndarray,
                                       hX: float, hT: float, hR: float, conf_level: float,
                                       target_time_index: float) -> float:  # 估计加权 split conformal 的 qhat
    build = batch_tbl_fixed.iloc[build_ids].copy()  # 取训练集 build set
    calib = batch_tbl_fixed.iloc[calib_ids].copy()  # 取校准集 calibration set

    if calib.shape[0] == 0:  # 若校准集为空
        return 0.0  # 返回 0，避免后续报错

    prior_new_build = prior_new_from_proxy(R_grid, build, hR)  # 基于训练集构造最近信息先验
    delta = R_grid[1] - R_grid[0]  # 计算网格步长

    phat = np.zeros(calib.shape[0], dtype=float)  # 初始化校准集的预测失效率数组
    ptrue = calib["p_true_local"].to_numpy(dtype=float)  # 取校准集的局部真失效率代理值

    for i in range(calib.shape[0]):  # 逐个校准批次计算预测失效率
        x_i = int(calib["X_ppm"].iloc[i])  # 取当前校准批次的 ppm

        prior_hist_xi = prior_hist_given_x(  # 基于 build set 构造该校准批次的历史条件先验
            R_grid=R_grid,
            batch_tbl=build,
            x_ppm=x_i,
            target_time_index=target_time_index,
            hX=hX, hT=hT, hR=hR
        )

        prior_xi = mix_prior(prior_hist_xi, prior_new_build, R_grid, alpha=alpha_now)  # 混合得到当前校准批次的最终先验
        ER = (R_grid * prior_xi).sum() * delta  # 计算先验下 E[R]
        phat[i] = 1.0 - ER  # 预测失效率 = 1 - E[R]

    resid = ptrue - phat  # 计算校准残差

    # 这次不截断 qhat，所以这里只返回原始 qhat
    qhat = float(np.quantile(resid, conf_level, method="linear"))  # 取残差的 conf_level 分位数作为 qhat
    return qhat  # 返回原始 qhat


# =========================================================
# Core runner for one (ppm2025, LTPD_ppm)
# =========================================================

def run_one_setting(batch_tbl_fixed: pd.DataFrame,
                    ppm_2025: int,
                    ltpd_ppm: int,
                    safe_out_dir: str = None,
                    save_rep: bool = False):
    """
    对一个给定的：
      - 2025 最新产品 ppm
      - 客户 LTPD (ppm)
    运行 N_REP 次 replication，返回结果摘要
    """

    R_grid = np.arange(0.0, 1.0001, 0.001)  # 构造可靠率积分网格

    proxy_rows = batch_tbl_fixed[  # 找到当前目标批次代理输入对应的历史行
        (batch_tbl_fixed["类别"].astype(str) == str(NEW_PROXY_CLASS)) &
        (batch_tbl_fixed["段"] == NEW_PROXY_SEG)
    ]

    if proxy_rows.shape[0] == 0:  # 若找不到代理行
        raise RuntimeError(
            f"Cannot find proxy rows for NEW_PROXY_CLASS={NEW_PROXY_CLASS}, NEW_PROXY_SEG={NEW_PROXY_SEG}."
        )  # 抛出错误，提示代理批次设置有问题

    x_new_ppm = int(proxy_rows["X_ppm"].iloc[0])  # 读取代理输入对应的 ppm
    if XNEW_TIMES_10:  # 若启用了 ×10 口径
        x_new_ppm = int(10 * x_new_ppm)  # 则把代理 ppm 放大 10 倍

    TARGET_TIME_INDEX = len(HIST_CLASS_NAMES) * SEG_K  # 目标批次位于历史时间链条之后的下一个位置

    prior_hist_xnew_fixed = prior_hist_given_x(  # 构造当前代理输入下的历史条件先验
        R_grid, batch_tbl_fixed,
        x_ppm=x_new_ppm,
        target_time_index=TARGET_TIME_INDEX,
        hX=HX_PPM, hT=HT_TIME, hR=hR
    )

    prior_new_fixed = prior_new_from_proxy(R_grid, batch_tbl_fixed, hR=hR)  # 构造最近参考批次先验

    rng_split = np.random.default_rng(SPLIT_SEED_FIXED)  # 初始化固定划分随机数生成器
    id_all = rng_split.permutation(batch_tbl_fixed.shape[0])  # 随机打乱历史批次摘要行索引

    n_build = max(1, int(np.floor(BUILD_FRAC * batch_tbl_fixed.shape[0])))  # 计算 build set 大小
    build_ids = id_all[:n_build]  # 训练集索引
    calib_ids = id_all[n_build:]  # 校准集索引

    if calib_ids.size == 0:  # 若校准集为空
        raise RuntimeError("Calibration split empty. Reduce BUILD_FRAC or increase data.")  # 提示需要调整数据量或 BUILD_FRAC

    rep_rows = []  # 用于收集所有 replication 的结果
    alpha_t = alpha_init  # 初始化在线更新历史权重 alpha
    err_hist = []  # 用于保存历轮误差，供窗口平滑使用

    # 这里按当前设定的 2025 真值 ppm 生成测试池
    NEW_p_true_2025 = ppm_2025 / 1e6  # 把当前 2025 真值 ppm 转成失效率

    # 当前客户 LTPD
    P_BAD = ltpd_ppm / 1e6  # 把当前客户 LTPD 从 ppm 转成概率

    # 仍保留你原来的 p_prior_t 逻辑，不额外大改
    x_new_ppm_base = int(x_new_ppm / 10) if XNEW_TIMES_10 else int(x_new_ppm)  # 若前面放大过 10 倍，则这里还原
    p_prior_fixed_from_xnew = (x_new_ppm_base / 1e6)  # 把代理输入 ppm 直接转成基准先验失效率

    ERR_SCALE_LOCAL = max(NEW_p_true_2025, 1e-6)  # 构造误差标准化尺度，避免除零

    for t in range(1, N_REP + 1):  # 逐轮执行 replication
        seed_2025 = SEED_BASE + 999999 + t  # 为当前轮构造独立随机种子
        alpha_before = alpha_t  # 记录当前轮更新前的 alpha

        prior_dens_xnew = mix_prior(prior_hist_xnew_fixed, prior_new_fixed, R_grid, alpha=alpha_before)  # 用当前 alpha 混合先验

        qhat_raw = estimate_qhat_weighted_split_fixed(  # 估计当前 alpha 下的 qhat
            batch_tbl_fixed=batch_tbl_fixed,
            build_ids=build_ids,
            calib_ids=calib_ids,
            alpha_now=alpha_before,
            R_grid=R_grid,
            hX=HX_PPM, hT=HT_TIME, hR=hR,
            conf_level=conf_level,
            target_time_index=TARGET_TIME_INDEX
        )

        # 这次不做 max(qhat_raw, 0)
        qhat_use = qhat_raw  # 直接使用原始 qhat

        N_star = solve_Nstar_by_postconf(  # 根据后验置信条件反解 N*
            prior_dens=prior_dens_xnew,
            R_grid=R_grid,
            qhat=qhat_use,
            p_bad=P_BAD,
            conf_target=CONF_TARGET,
            N_max=N_max
        )

        p_bad_eff_now = clamp01(P_BAD - qhat_use)  # 计算当前 replication 下经 qhat 修正后的有效坏品率门槛
        posterior_conf_at_nstar = posterior_conf_good(N_star, prior_dens_xnew, R_grid, p_bad_eff_now) if N_star is not None else np.nan  # 计算 N* 对应的后验好批次置信度
        post_dens_zero_fail = posterior_density_zero_fail(N_star, prior_dens_xnew, R_grid) if N_star is not None else np.nan  # 在零失效条件下构造 N* 对应的后验分布
        posterior_mean_R = posterior_mean_R_from_density(post_dens_zero_fail, R_grid) if N_star is not None else np.nan  # 计算后验平均可靠率
        posterior_mean_p = float(np.clip(1.0 - posterior_mean_R, 0.0, 1.0)) if N_star is not None and not np.isnan(posterior_mean_R) else np.nan  # 计算后验平均失效率

        df_2025_pool = simulate_2025_pool(  # 生成当前 2025 真值 ppm 下的测试池
            seed=seed_2025,
            n_batches=Y25_N_BATCHES_POOL,
            per_batch=Y25_PER_BATCH_POOL,
            p_true=NEW_p_true_2025
        )

        # 新增：再用当前 replication 下的后验分布生成一个新的 2025 测试池，用来检验同一个 N* 在“后验生成池子”中的通过率表现
        if N_star is not None:
            df_2025_pool_post, R_post_draw, p_post_draw = simulate_2025_pool_from_posterior(
                seed=seed_2025 + 333333,
                n_batches=Y25_N_BATCHES_POOL,
                per_batch=Y25_PER_BATCH_POOL,
                post_dens=post_dens_zero_fail,
                R_grid=R_grid
            )  # 用后验分布生成新的 2025 测试池
        else:
            df_2025_pool_post, R_post_draw, p_post_draw = None, np.nan, np.nan  # 若 N* 无效，则后验生成池子也置为空

        test_n = df_2025_pool.shape[0]  # 当前测试池总样本量
        replace_flag = False  # 默认无放回抽样

        if N_star is not None and N_star > test_n:  # 若 N* 超过测试池大小
            if TEST_SAMPLE_REPLACE_IF_NEEDED:  # 若允许有放回
                replace_flag = True  # 切换为有放回抽样
            else:  # 若不允许有放回
                N_star = None  # 当前轮视为无法抽样

        F_t = None  # 初始化当前轮失效个数
        p_obs_t = np.nan  # 初始化当前轮观测失效率
        passed = None  # 初始化当前轮是否通过
        F_t_post = None  # 新增：初始化后验生成池子的失效个数
        p_obs_t_post = np.nan  # 新增：初始化后验生成池子的观测失效率
        passed_post = None  # 新增：初始化后验生成池子的通过结果

        if N_star is not None and N_star > 0:  # 若当前轮得到有效 N*
            rng = np.random.default_rng(seed_2025 + 777)  # 初始化抽样随机数生成器
            idx = rng.choice(test_n, size=N_star, replace=replace_flag)  # 从真实 2025 测试池抽取 N* 个样本
            F_t = int(df_2025_pool["是否失效"].to_numpy()[idx].sum())  # 统计真实池子的失效个数
            p_obs_t = F_t / N_star  # 计算真实池子的观测失效率
            passed = int(pass_rule_test(F_t))  # 按零失效规则判断真实池子是否通过

            test_n_post = df_2025_pool_post.shape[0]  # 读取后验生成测试池的总样本量
            replace_flag_post = bool(TEST_SAMPLE_REPLACE_IF_NEEDED and N_star > test_n_post)  # 若样本量超过后验池大小，则按原规则决定是否有放回抽样
            rng_post = np.random.default_rng(seed_2025 + 888)  # 初始化后验生成池子的抽样随机数生成器
            idx_post = rng_post.choice(test_n_post, size=N_star, replace=replace_flag_post)  # 从后验生成的测试池抽取 N* 个样本
            F_t_post = int(df_2025_pool_post["是否失效"].to_numpy()[idx_post].sum())  # 统计后验生成池子的失效个数
            p_obs_t_post = F_t_post / N_star  # 计算后验生成池子的观测失效率
            passed_post = int(pass_rule_test(F_t_post))  # 按零失效规则判断后验生成池子是否通过

        pass_prob = prior_pass_prob(N_star, prior_dens_xnew, R_grid) if N_star is not None else np.nan  # 先验下零失效通过概率
        p_prior_imp = p_prior_implied(N_star, pass_prob) if N_star is not None else np.nan  # 反推出等效先验失效率

        p_prior_t = p_prior_fixed_from_xnew * 10.0  # 保留你原来的先验基准失效率口径
        err_t = (p_obs_t - p_prior_t) if not np.isnan(p_obs_t) else np.nan  # 当前轮误差 = 观测失效率 - 基准先验失效率

        if not np.isnan(err_t):  # 若当前轮误差有效
            err_hist.append(err_t)  # 加入误差历史列表

        err_smooth = np.nan  # 初始化平滑误差
        if len(err_hist) > 0:  # 若已有误差历史
            D = min(WIN, len(err_hist))  # 窗口长度取 WIN 与当前误差数的较小者
            err_smooth = float(np.mean(err_hist[-D:]))  # 计算最近 D 轮的平均误差

            delta_alpha = clamp_step(STEP_SIZE * (err_smooth / ERR_SCALE_LOCAL))  # 计算并截断 alpha 更新量
            alpha_t = clamp_alpha(alpha_t - delta_alpha)  # 更新 alpha 并裁剪到允许区间

        alpha_after = alpha_t  # 记录当前轮更新后的 alpha

        rep_rows.append({  # 保存当前轮的所有结果
            "rep_id": t,
            "ppm_2025": ppm_2025,
            "ltpd_ppm": ltpd_ppm,
            "alpha_before": alpha_before,
            "alpha_after": alpha_after,
            "N_star": N_star,
            "F_t": F_t,
            "p_obs_t": p_obs_t,
            "p_prior_imp": p_prior_imp,
            "p_prior_xnew": p_prior_t,
            "posterior_conf_at_nstar": posterior_conf_at_nstar,
            "posterior_mean_R": posterior_mean_R,
            "posterior_mean_p": posterior_mean_p,
            "posterior_draw_R": R_post_draw,
            "posterior_draw_p": p_post_draw,
            "F_t_post": F_t_post,
            "p_obs_t_post": p_obs_t_post,
            "pass_post": passed_post,
            "err_t": err_t,
            "err_smooth": err_smooth,
            "pass": passed,
            "qhat_raw": qhat_raw,
            "qhat_used": qhat_use
        })

    rep_df = pd.DataFrame(rep_rows)  # 汇总所有 replication 结果

    if save_rep and safe_out_dir is not None:  # 若要求保存 replication 级结果
        fn_rep = os.path.join(
            safe_out_dir,
            f"rep_ppm2025_{ppm_2025}_ltpd_{ltpd_ppm}.csv"
        )  # 构造输出文件名
        rep_df.to_csv(fn_rep, index=False)  # 导出 CSV

    pass_rate = float(rep_df["pass"].dropna().mean()) if rep_df["pass"].notna().any() else np.nan  # 计算真实 2025 测试池上的通过率
    pass_rate_post = float(rep_df["pass_post"].dropna().mean()) if rep_df["pass_post"].notna().any() else np.nan  # 新增：计算后验生成测试池上的通过率
    posterior_conf_mean = float(rep_df["posterior_conf_at_nstar"].dropna().mean()) if rep_df["posterior_conf_at_nstar"].notna().any() else np.nan  # 新增：计算 N* 对应后验置信度的平均值

    out = {  # 构造该组合的摘要输出
        "2025_ppm": ppm_2025,
        "LTPD_ppm": ltpd_ppm,
        "N1_star": int(rep_df.loc[rep_df["rep_id"] == 1, "N_star"].iloc[0]) if rep_df["N_star"].notna().any() else np.nan,
        "N100_star": int(rep_df.loc[rep_df["rep_id"] == N_REP, "N_star"].iloc[0]) if rep_df["N_star"].notna().any() else np.nan,
        "Nstar_mean": float(rep_df["N_star"].dropna().mean()) if rep_df["N_star"].notna().any() else np.nan,
        "Nstar_min": int(rep_df["N_star"].dropna().min()) if rep_df["N_star"].notna().any() else np.nan,
        "Nstar_max": int(rep_df["N_star"].dropna().max()) if rep_df["N_star"].notna().any() else np.nan,
        "posterior_conf_mean": posterior_conf_mean,
        "pass_rate": pass_rate,
        "post_pool_pass_rate": pass_rate_post,
        "producer_risk": 1.0 - pass_rate if pd.notna(pass_rate) else np.nan,
        "consumer_risk": pass_rate if pd.notna(pass_rate) else np.nan,
        "producer_risk_post": 1.0 - pass_rate_post if pd.notna(pass_rate_post) else np.nan,
        "consumer_risk_post": pass_rate_post if pd.notna(pass_rate_post) else np.nan
    }

    return out, rep_df  # 返回当前 (ppm2025, LTPD) 组合的摘要和逐轮结果


def run_posterior_only_setting(batch_tbl_fixed: pd.DataFrame,
                               ltpd_ppm: int,
                               safe_out_dir: str = None,
                               save_rep: bool = False):  # 固定 x_new 与历史数据，仅用后验生成池子来验证 N*
    """
    在该函数中：
      1) 不再使用真实 2025 ppm 生成测试池；
      2) 只固定 x_new=代理输入与历史数据；
      3) 用当前 replication 下的后验分布生成新的 2025 测试池；
      4) 再用同一个 N* 去验证后验生成池的通过率。
    """

    R_grid = np.arange(0.0, 1.0001, 0.001)  # 构造可靠率积分网格

    proxy_rows = batch_tbl_fixed[  # 找到当前目标批次代理输入对应的历史行
        (batch_tbl_fixed["类别"].astype(str) == str(NEW_PROXY_CLASS)) &
        (batch_tbl_fixed["段"] == NEW_PROXY_SEG)
    ]

    if proxy_rows.shape[0] == 0:  # 若找不到代理行
        raise RuntimeError(
            f"Cannot find proxy rows for NEW_PROXY_CLASS={NEW_PROXY_CLASS}, NEW_PROXY_SEG={NEW_PROXY_SEG}."
        )

    x_new_ppm = int(proxy_rows["X_ppm"].iloc[0])  # 读取代理输入对应的 ppm
    if XNEW_TIMES_10:  # 若启用 ×10 口径
        x_new_ppm = int(10 * x_new_ppm)  # 则把代理 ppm 放大 10 倍

    TARGET_TIME_INDEX = len(HIST_CLASS_NAMES) * SEG_K  # 目标批次位于历史时间链条之后的下一个位置

    prior_hist_xnew_fixed = prior_hist_given_x(  # 构造当前代理输入下的历史条件先验
        R_grid, batch_tbl_fixed,
        x_ppm=x_new_ppm,
        target_time_index=TARGET_TIME_INDEX,
        hX=HX_PPM, hT=HT_TIME, hR=hR
    )

    prior_new_fixed = prior_new_from_proxy(R_grid, batch_tbl_fixed, hR=hR)  # 构造最近参考批次先验

    rng_split = np.random.default_rng(SPLIT_SEED_FIXED)  # 初始化固定划分随机数生成器
    id_all = rng_split.permutation(batch_tbl_fixed.shape[0])  # 随机打乱历史批次摘要索引

    n_build = max(1, int(np.floor(BUILD_FRAC * batch_tbl_fixed.shape[0])))  # 计算训练集大小
    build_ids = id_all[:n_build]  # 训练集索引
    calib_ids = id_all[n_build:]  # 校准集索引

    if calib_ids.size == 0:  # 若校准集为空
        raise RuntimeError("Calibration split empty. Reduce BUILD_FRAC or increase data.")

    rep_rows = []  # 用于保存所有 posterior-only replication 的结果
    alpha_t = alpha_init  # 初始化在线更新权重 alpha
    err_hist = []  # 保存误差历史，供窗口平滑使用

    P_BAD = ltpd_ppm / 1e6  # 当前客户 LTPD 转成坏品率概率

    x_new_ppm_base = int(x_new_ppm / 10) if XNEW_TIMES_10 else int(x_new_ppm)  # 若前面放大过 10 倍，则这里还原基础代理 ppm
    p_prior_fixed_from_xnew = (x_new_ppm_base / 1e6)  # 把代理输入 ppm 直接转成基准先验失效率

    ERR_SCALE_LOCAL = max(p_prior_fixed_from_xnew, 1e-6)  # posterior-only 场景下用代理输入对应的先验失效率做误差标准化尺度

    for t in range(1, N_REP + 1):  # 逐轮 posterior-only replication
        seed_post = SEED_BASE + 2020000 + int(ltpd_ppm) * 100 + t  # 为当前 posterior-only replication 构造独立随机种子
        alpha_before = alpha_t  # 记录当前轮更新前的 alpha

        prior_dens_xnew = mix_prior(prior_hist_xnew_fixed, prior_new_fixed, R_grid, alpha=alpha_before)  # 用当前 alpha 混合先验

        qhat_raw = estimate_qhat_weighted_split_fixed(  # 估计当前 alpha 下的 qhat
            batch_tbl_fixed=batch_tbl_fixed,
            build_ids=build_ids,
            calib_ids=calib_ids,
            alpha_now=alpha_before,
            R_grid=R_grid,
            hX=HX_PPM, hT=HT_TIME, hR=hR,
            conf_level=conf_level,
            target_time_index=TARGET_TIME_INDEX
        )

        qhat_use = qhat_raw  # 继续直接使用原始 qhat

        N_star = solve_Nstar_by_postconf(  # 根据后验置信条件反解当前轮 N*
            prior_dens=prior_dens_xnew,
            R_grid=R_grid,
            qhat=qhat_use,
            p_bad=P_BAD,
            conf_target=CONF_TARGET,
            N_max=N_max
        )

        p_bad_eff_now = clamp01(P_BAD - qhat_use)  # 构造当前轮有效坏品率阈值
        posterior_conf_at_nstar = posterior_conf_good(N_star, prior_dens_xnew, R_grid, p_bad_eff_now) if N_star is not None else np.nan  # 计算当前 N* 对应的后验置信度

        post_dens_at_nstar = posterior_density_zero_fail(N_star, prior_dens_xnew, R_grid)  # 在零失效条件下构造当前 N* 对应的后验密度
        posterior_mean_R = posterior_mean_R_from_density(post_dens_at_nstar, R_grid) if N_star is not None else np.nan  # 计算当前 N* 对应的后验平均可靠率
        posterior_mean_p = (1.0 - posterior_mean_R) if pd.notna(posterior_mean_R) else np.nan  # 计算当前 N* 对应的后验平均失效率

        df_post_pool, R_post_draw, p_post_draw = simulate_2025_pool_from_posterior(  # 用当前后验分布生成新的 2025 测试池
            seed=seed_post,
            n_batches=Y25_N_BATCHES_POOL,
            per_batch=Y25_PER_BATCH_POOL,
            post_dens=post_dens_at_nstar,
            R_grid=R_grid
        )

        test_n_post = df_post_pool.shape[0]  # posterior 生成测试池总样本量
        replace_flag_post = False  # 默认无放回抽样

        if N_star is not None and N_star > test_n_post:  # 若 N* 超过 posterior 测试池总量
            if TEST_SAMPLE_REPLACE_IF_NEEDED:  # 若允许有放回抽样
                replace_flag_post = True  # 切换为有放回抽样
            else:  # 若不允许有放回
                N_star = None  # 当前轮视为无法抽样

        F_t_post = None  # 初始化 posterior 生成池下的失效数
        p_obs_t_post = np.nan  # 初始化 posterior 生成池下的观测失效率
        passed_post = None  # 初始化 posterior 生成池下的是否通过

        if N_star is not None and N_star > 0:  # 若当前轮得到有效 N*
            rng_post = np.random.default_rng(seed_post + 777)  # 初始化 posterior 生成池的抽样随机数生成器
            idx_post = rng_post.choice(test_n_post, size=N_star, replace=replace_flag_post)  # 从 posterior 生成池抽取 N* 个样本
            F_t_post = int(df_post_pool["是否失效"].to_numpy()[idx_post].sum())  # 统计 posterior 生成池下的失效个数
            p_obs_t_post = F_t_post / N_star  # 计算 posterior 生成池下的观测失效率
            passed_post = int(pass_rule_test(F_t_post))  # 按零失效规则判断 posterior 生成池是否通过

        p_prior_t = p_prior_fixed_from_xnew * 10.0  # 保留原有先验基准失效率口径
        err_t_post = (p_obs_t_post - p_prior_t) if not np.isnan(p_obs_t_post) else np.nan  # 用 posterior 生成池观测失效率计算当前轮误差

        if not np.isnan(err_t_post):  # 若当前轮误差有效
            err_hist.append(err_t_post)  # 加入误差历史

        err_smooth = np.nan  # 初始化平滑误差
        if len(err_hist) > 0:  # 若已有误差历史
            D = min(WIN, len(err_hist))  # 窗口长度取 WIN 与当前误差数的较小者
            err_smooth = float(np.mean(err_hist[-D:]))  # 计算最近 D 轮的平均误差

            delta_alpha = clamp_step(STEP_SIZE * (err_smooth / ERR_SCALE_LOCAL))  # 计算并截断 alpha 更新量
            alpha_t = clamp_alpha(alpha_t - delta_alpha)  # 更新 alpha 并裁剪到允许区间

        alpha_after = alpha_t  # 记录当前轮更新后的 alpha

        rep_rows.append({  # 保存当前轮 posterior-only 结果
            "rep_id": t,
            "ltpd_ppm": ltpd_ppm,
            "alpha_before": alpha_before,
            "alpha_after": alpha_after,
            "N_star": N_star,
            "posterior_conf_at_nstar": posterior_conf_at_nstar,
            "posterior_mean_R": posterior_mean_R,
            "posterior_mean_p": posterior_mean_p,
            "posterior_draw_R": R_post_draw,
            "posterior_draw_p": p_post_draw,
            "F_t_post": F_t_post,
            "p_obs_t_post": p_obs_t_post,
            "pass_post": passed_post,
            "err_t_post": err_t_post,
            "err_smooth": err_smooth,
            "qhat_raw": qhat_raw,
            "qhat_used": qhat_use
        })

    rep_df = pd.DataFrame(rep_rows)  # 汇总 posterior-only replication 结果

    if save_rep and safe_out_dir is not None:  # 若要求保存 posterior-only replication 级结果
        fn_rep = os.path.join(safe_out_dir, f"posterior_only_rep_ltpd_{ltpd_ppm}.csv")  # 构造 posterior-only 输出文件名
        rep_df.to_csv(fn_rep, index=False)  # 导出 CSV

    pass_rate_post = float(rep_df["pass_post"].dropna().mean()) if rep_df["pass_post"].notna().any() else np.nan  # 计算 posterior 生成池上的通过率
    posterior_conf_mean = float(rep_df["posterior_conf_at_nstar"].dropna().mean()) if rep_df["posterior_conf_at_nstar"].notna().any() else np.nan  # 计算 N* 对应后验置信度的平均值

    out = {  # 构造 posterior-only 摘要输出
        "LTPD_ppm": ltpd_ppm,
        "N1_star": int(rep_df.loc[rep_df["rep_id"] == 1, "N_star"].iloc[0]) if rep_df["N_star"].notna().any() else np.nan,
        "N100_star": int(rep_df.loc[rep_df["rep_id"] == N_REP, "N_star"].iloc[0]) if rep_df["N_star"].notna().any() else np.nan,
        "posterior_conf_at_nstar": posterior_conf_mean,
        "posterior_pool_pass_rate": pass_rate_post
    }

    return out, rep_df  # 返回 posterior-only 摘要结果与逐轮结果


# =========================================================
# Main program
# =========================================================

def main():  # 主程序入口
    safe_out_dir = pick_writable_dir()  # 选择可写输出目录
    print(f"\nFiles will be saved to: {safe_out_dir}\n")  # 打印输出目录

    # Step 1: 先构造弱波动历史数据，并按公司要求的输入格式导出 Excel / CSV
    ppm_table_fixed_sim = build_fixed_ppm_table()  # 生成弱波动历史 ppm 场景表

    seed_hist_fixed_sim = SEED_BASE + 1  # 历史数据模拟种子
    df_hist_fixed_sim = simulate_hist_2020_2024(  # 生成历史样本级数据
        seed_hist_fixed_sim,
        ppm_table_fixed_sim,
        HIST_PER_SEG_BATCH,
        N_PER_SEG_BATCH
    )
    batch_tbl_fixed_sim = summarise_batches(df_hist_fixed_sim)  # 汇总成历史批次摘要表（模拟版）

    if GENERATE_SIM_COMPANY_INPUT:  # 若需要先导出公司格式输入文件
        export_simulation_to_company_format(batch_tbl_fixed_sim, REAL_INPUT_PATH)  # 导出公司输入格式 Excel / CSV

    if USE_REAL_INPUT:  # 若主程序后续优先从公司格式文件读取
        batch_tbl_fixed = load_real_batch_table(REAL_INPUT_PATH, REAL_INPUT_SHEET)  # 按公司格式重新读回历史批次表
    else:  # 否则仍直接使用模拟汇总表
        batch_tbl_fixed = batch_tbl_fixed_sim  # 保持原有内部模拟逻辑

    # 保存历史输入表，便于检查
    hist_input_path = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_historical_input.csv")  # 构造历史输入表路径
    batch_tbl_fixed.to_csv(hist_input_path, index=False, encoding="utf-8-sig")  # 导出历史输入表

    # Step 2: 跑所有 (ppm2025, LTPD) 组合
    all_summary = []  # 用于保存所有组合的摘要结果

    total_jobs = len(PPM_2025_GRID_ALL) * len(LTPD_PPM_GRID)  # 计算总任务数
    job_id = 0  # 当前任务编号

    for ppm_2025 in PPM_2025_GRID_ALL:  # 遍历所有 2025 ppm
        for ltpd_ppm in LTPD_PPM_GRID:  # 遍历所有客户 LTPD
            job_id += 1  # 当前任务编号加一
            print(f"[{job_id}/{total_jobs}] Running ppm_2025={ppm_2025}, LTPD_ppm={ltpd_ppm} ...")  # 打印当前运行信息

            summary_row, _ = run_one_setting(  # 运行该组合
                batch_tbl_fixed=batch_tbl_fixed,
                ppm_2025=ppm_2025,
                ltpd_ppm=ltpd_ppm,
                safe_out_dir=safe_out_dir,
                save_rep=False
            )
            all_summary.append(summary_row)  # 保存摘要结果

    summary_df = pd.DataFrame(all_summary)  # 把全部组合结果整理成长表

    # 保存完整长表
    fn_summary = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_summary_long.csv")  # 完整长表路径
    summary_df.to_csv(fn_summary, index=False, encoding="utf-8-sig")  # 导出完整长表

    # Step 3: 生成 N1 宽表
    n1_wide = summary_df.pivot(index="2025_ppm", columns="LTPD_ppm", values="N1_star").reset_index()  # 构造第 1 次 N* 宽表
    fn_n1 = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_N1_wide.csv")  # N1 宽表路径
    n1_wide.to_csv(fn_n1, index=False, encoding="utf-8-sig")  # 导出 N1 宽表

    # Step 4: 生成 N100 宽表
    n100_wide = summary_df.pivot(index="2025_ppm", columns="LTPD_ppm", values="N100_star").reset_index()  # 构造第 100 次 N* 宽表
    fn_n100 = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_N100_wide.csv")  # N100 宽表路径
    n100_wide.to_csv(fn_n100, index=False, encoding="utf-8-sig")  # 导出 N100 宽表

    # Step 5: 生成 pass_rate 宽表
    pass_wide = summary_df.pivot(index="2025_ppm", columns="LTPD_ppm", values="pass_rate").reset_index()  # 构造真实 2025 测试池的通过率宽表
    fn_pass = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_passrate_wide.csv")  # 通过率宽表路径
    pass_wide.to_csv(fn_pass, index=False, encoding="utf-8-sig")  # 导出通过率宽表

    # Step 5b: 新增“纯 posterior validation”摘要表
    posterior_only_rows = []  # 用于保存纯 posterior validation 摘要结果
    for ltpd_ppm in LTPD_PPM_GRID:  # 逐个 LTPD 运行 posterior-only 验证
        print(f"[posterior-only] Running LTPD_ppm={ltpd_ppm} ...")  # 打印 posterior-only 运行信息
        posterior_row, _ = run_posterior_only_setting(  # 固定 x_new 和历史数据，仅用后验生成池子来验证 N*
            batch_tbl_fixed=batch_tbl_fixed,
            ltpd_ppm=ltpd_ppm,
            safe_out_dir=safe_out_dir,
            save_rep=False
        )
        posterior_only_rows.append(posterior_row)  # 保存 posterior-only 摘要结果

    posterior_only_df = pd.DataFrame(posterior_only_rows)  # 整理纯 posterior validation 摘要表
    posterior_only_df = posterior_only_df[["LTPD_ppm", "N1_star", "N100_star", "posterior_conf_at_nstar", "posterior_pool_pass_rate"]].copy()  # 按公司要求保留固定列顺序
    fn_posterior_only = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_posterior_only_validation.csv")  # 构造纯 posterior validation 摘要表路径
    posterior_only_df.to_csv(fn_posterior_only, index=False, encoding="utf-8-sig")  # 导出纯 posterior validation 摘要表

    # Step 6: 针对某一个固定 LTPD，做公司喜欢看的“好批次 / 坏批次摘要表”
    selected_df = summary_df[summary_df["LTPD_ppm"] == SELECTED_LTPD_PPM_FOR_SUMMARY].copy()  # 筛出指定 LTPD 的结果

    good_df = selected_df[selected_df["2025_ppm"].isin(PPM_2025_GRID_GOOD)].copy()  # 提取好批次部分
    good_df = good_df.sort_values("2025_ppm").reset_index(drop=True)  # 按 ppm 排序
    good_df["好批次通过率=通过率"] = good_df["pass_rate"]  # 添加真实测试池下的好批次通过率列
    good_df["生产者风险=1-好批次通过率"] = 1.0 - good_df["pass_rate"]  # 添加真实测试池下的生产者风险列

    good_out = good_df[[
        "2025_ppm", "N1_star", "N100_star",
        "好批次通过率=通过率", "生产者风险=1-好批次通过率"
    ]].copy()  # 选择公司摘要表需要的列
    good_out = good_out.rename(columns={"2025_ppm": "2025 年批次产片的全生命周期ppm（好批次）"})  # 改为中文列名
    fn_good = os.path.join(
        safe_out_dir,
        f"{OUTPUT_BASENAME}_good_batches_LTPD_{SELECTED_LTPD_PPM_FOR_SUMMARY}.csv"
    )  # 构造好批次摘要表路径
    good_out.to_csv(fn_good, index=False, encoding="utf-8-sig")  # 导出好批次摘要表

    bad_df = selected_df[selected_df["2025_ppm"].isin(PPM_2025_GRID_BAD)].copy()  # 提取坏批次部分
    bad_df = bad_df.sort_values("2025_ppm").reset_index(drop=True)  # 按 ppm 排序
    bad_df["坏批次通过率=通过率"] = bad_df["pass_rate"]  # 添加真实测试池下的坏批次通过率列
    bad_df["消费者风险=坏批次通过率"] = bad_df["pass_rate"]  # 添加真实测试池下的消费者风险列

    bad_out = bad_df[[
        "2025_ppm", "N1_star", "N100_star",
        "坏批次通过率=通过率", "消费者风险=坏批次通过率"
    ]].copy()  # 选择公司摘要表需要的列
    bad_out = bad_out.rename(columns={"2025_ppm": "2025 年批次产片的全生命周期ppm（坏批次）"})  # 改为中文列名
    fn_bad = os.path.join(
        safe_out_dir,
        f"{OUTPUT_BASENAME}_bad_batches_LTPD_{SELECTED_LTPD_PPM_FOR_SUMMARY}.csv"
    )  # 构造坏批次摘要表路径
    bad_out.to_csv(fn_bad, index=False, encoding="utf-8-sig")  # 导出坏批次摘要表

    # Step 7: 打印结果说明
    print("\n===== DONE =====")  # 打印完成标记
    print("历史输入表:", hist_input_path)  # 打印历史输入表路径
    print("完整长表:", fn_summary)  # 打印完整长表路径
    print("N1 宽表:", fn_n1)  # 打印 N1 宽表路径
    print("N100 宽表:", fn_n100)  # 打印 N100 宽表路径
    print("通过率宽表:", fn_pass)  # 打印真实测试池通过率宽表路径
    print("纯 posterior validation 表:", fn_posterior_only)  # 打印后验生成测试池通过率宽表路径
    print("好批次摘要表:", fn_good)  # 打印好批次摘要表路径
    print("坏批次摘要表:", fn_bad)  # 打印坏批次摘要表路径

    print("\n===== 示例预览：好批次摘要表 =====")  # 打印好批次摘要表预览标题
    print(good_out.head(20))  # 预览好批次摘要表前 20 行

    print("\n===== 示例预览：坏批次摘要表 =====")  # 打印坏批次摘要表预览标题
    print(bad_out.head(20))  # 预览坏批次摘要表前 20 行


if __name__ == "__main__":  # 如果当前文件作为主程序执行
    main()  # 运行主函数
