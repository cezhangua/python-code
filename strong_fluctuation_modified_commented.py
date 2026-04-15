import os  # 导入 os 模块
import tempfile  # 导入 tempfile 模块
import numpy as np  # 导入 numpy 并记为 np
import pandas as pd  # 导入 pandas 并记为 pd
from scipy.stats import norm  # 导入正态分布函数 norm


# =========================================================
# Global params
# =========================================================

LTPD_TOTAL_DEFAULT = 0.01  # 设置默认 LTPD 占位值
conf_level = 0.90  # 设置 Conformal 分位数水平
hR = 0.02  # 设置可靠率方向带宽
N_max = 6000  # 设置搜索样本量上限

r_allow = 0  # 设置允许失效数
def pass_rule_test(x_fail: int) -> bool:  # 定义函数 pass_rule_test
    return x_fail <= r_allow  # 返回结果

N_REP = 100  # 设置重复实验次数
SEED_BASE = 20250101  # 设置基础随机种子

# =========================================================
# 强波动场景：跨年份 base ppm
# 整体下降，但波动更明显
# =========================================================
HIST_CLASS_NAMES = ["2020", "2021", "2022", "2023", "2024"]  # 设置历史年份类别
HIST_CLASS_DPPM = np.array([11000, 7200, 4800, 2600, 900], dtype=float)  # 设置各年份基础 ppm

SEG_K = 5  # 设置每年分段数

# =========================================================
# 强波动场景：年内 5 段明显震荡
# 非单调，有回升有下降，但总体仍偏下降
# =========================================================
SEG_FACTOR = np.array([1.35, 0.92, 1.18, 0.78, 0.62], dtype=float)  # 设置年内分段波动因子
SEG_FACTOR = SEG_FACTOR / SEG_FACTOR.mean()  # 把分段因子归一化到均值为 1

NEW_PROXY_CLASS = "2024"  # 设置代理年份
NEW_PROXY_SEG = 1  # 设置代理分段

Y25_N_BATCHES_POOL = 10  # 设置 2025 测试池批次数
Y25_PER_BATCH_POOL = 4000  # 设置 2025 每批样本数
TEST_SAMPLE_REPLACE_IF_NEEDED = True  # 设置样本不足时是否允许有放回抽样

# 强波动下可适当放大 ppm 带宽
HX_PPM = 500  # 设置 ppm 方向带宽

# 时间变量：年份 + 段 -> 顺序时间索引
BASE_YEAR = 2020  # 设置时间索引起始年份
HT_TIME = 5  # 设置时间方向带宽

alpha_init = 0.999  # 设置在线更新权重初值
ALPHA_MIN = 0.001  # 设置在线更新权重下界
ALPHA_MAX = 0.999  # 设置在线更新权重上界
STEP_SIZE = 1.0  # 设置在线更新步长系数
DELTA_MAX = 0.01  # 设置单轮最大更新幅度
WIN = 10  # 设置滑动窗口长度

BUILD_FRAC = 0.50  # 设置 build 集占比
SPLIT_SEED_FIXED = SEED_BASE + 55555  # 设置固定划分随机种子

HIST_PER_SEG_BATCH = 4000  # 设置每个历史分段批次样本量
N_PER_SEG_BATCH = 1  # 设置每个历史分段的批次数

CONF_TARGET = 0.90  # 设置目标后验置信度

# 不对 qhat 做 0 截断
QHAT_ONLY_TIGHTEN = False  # 设置 qhat 是否只向收紧方向处理

XNEW_TIMES_10 = True  # 设置是否把代理 ppm 放大 10 倍

OUTPUT_BASENAME = "strong_fluctuation_ltpd_grid"  # 设置输出文件名前缀


# =========================================================
# Company-format input switches
# =========================================================

GENERATE_SIM_COMPANY_INPUT = True  # 是否先把当前模拟历史数据导出成公司要求的 Excel 输入格式
USE_REAL_INPUT = True  # 后续主程序是否优先从公司格式 Excel / CSV 中读取历史输入
REAL_INPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "company_input_from_simulation_strong_fluctuation.xlsx")  # 公司格式输入文件路径（默认放在脚本同目录）
REAL_INPUT_SHEET = 0  # 若为 Excel，默认读取第 1 个 sheet


# =========================================================
# 用户可调网格
# =========================================================

PPM_2025_GRID_GOOD = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1400, 2000, 3000]  # 设置好批次 ppm 网格
PPM_2025_GRID_BAD  = [4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000]  # 设置坏批次 ppm 网格
PPM_2025_GRID_ALL = PPM_2025_GRID_GOOD + PPM_2025_GRID_BAD  # 合并全部 2025 ppm 网格

# LTPD 网格，已放到 10000
LTPD_PPM_GRID = [200, 400, 600, 800, 1000, 1400, 2000, 3000, 4000, 6000, 8000, 10000]  # 设置 LTPD 网格

# 生成公司摘要表时，选定一个客户 LTPD
SELECTED_LTPD_PPM_FOR_SUMMARY = 10000  # 设置摘要表使用的固定 LTPD


# =========================================================
# Utils
# =========================================================

def clamp01(x: float) -> float:  # 定义函数 clamp01
    return max(0.0, min(1.0, float(x)))  # 返回结果

def clamp_alpha(a: float) -> float:  # 定义函数 clamp_alpha
    return max(ALPHA_MIN, min(ALPHA_MAX, float(a)))  # 返回结果

def clamp_step(d: float) -> float:  # 定义函数 clamp_step
    return max(-DELTA_MAX, min(DELTA_MAX, float(d)))  # 返回结果

def make_time_index(year_str, seg, base_year=BASE_YEAR, seg_k=SEG_K):  # 定义函数 make_time_index
    year_num = int(year_str)  # 提取年份整数
    return (year_num - base_year) * seg_k + (int(seg) - 1)  # 返回结果

def kde1d_density(R_grid: np.ndarray, R_i: np.ndarray, hR: float) -> np.ndarray:  # 定义函数 kde1d_density
    delta = R_grid[1] - R_grid[0]  # 计算网格步长

    if R_i.size == 0:  # 条件判断
        dens = np.ones_like(R_grid, dtype=float)  # 构造密度数组
        return dens / (dens.sum() * delta)  # 返回结果

    z = (R_grid[:, None] - R_i[None, :]) / hR  # 计算标准化距离
    dens = norm.pdf(z) / hR  # 构造密度数组
    dens = dens.sum(axis=1)  # 构造密度数组
    dens = np.maximum(dens, 0.0)  # 构造密度数组
    dens = dens / (dens.sum() * delta)  # 构造密度数组
    return dens  # 返回结果

def kde1d_density_weighted(R_grid: np.ndarray, R_i: np.ndarray, w_i: np.ndarray, hR: float) -> np.ndarray:  # 定义函数 kde1d_density_weighted
    delta = R_grid[1] - R_grid[0]  # 计算网格步长

    if R_i.size == 0:  # 条件判断
        dens = np.ones_like(R_grid, dtype=float)  # 构造密度数组
        return dens / (dens.sum() * delta)  # 返回结果

    w = np.maximum(w_i.astype(float), 0.0)  # 构造权重
    if w.sum() <= 0:  # 条件判断
        w = np.ones_like(w)  # 构造权重

    z = (R_grid[:, None] - R_i[None, :]) / hR  # 计算标准化距离
    dens = (w[None, :] * (norm.pdf(z) / hR)).sum(axis=1)  # 构造密度数组
    dens = np.maximum(dens, 0.0)  # 构造密度数组
    dens = dens / (dens.sum() * delta)  # 构造密度数组
    return dens  # 返回结果

def posterior_conf_good(N: int, prior_dens: np.ndarray, R_grid: np.ndarray, p_bad_eff: float) -> float:  # 定义函数 posterior_conf_good
    if N is None or N <= 0:  # 条件判断
        return np.nan  # 返回结果

    delta = R_grid[1] - R_grid[0]  # 计算网格步长
    like = np.power(R_grid, N)  # 计算零失效似然
    post = prior_dens * like  # 计算未归一化后验
    Z = post.sum() * delta  # 计算归一化常数

    if Z <= 0:  # 条件判断
        return np.nan  # 返回结果

    R_thr = 1.0 - p_bad_eff  # 计算可靠率阈值
    idx = (R_grid >= R_thr)  # 定位满足条件的网格位置
    return (post[idx].sum() * delta) / Z  # 返回结果

def solve_Nstar_by_postconf(prior_dens: np.ndarray, R_grid: np.ndarray, qhat: float,  # 定义函数 solve_Nstar_by_postconf
                            p_bad: float, conf_target: float, N_max: int):  # 执行本行逻辑
    p_bad_eff = clamp01(p_bad - qhat)  # 执行本行逻辑

    for N in range(1, N_max + 1):  # 开始循环
        confN = posterior_conf_good(N, prior_dens, R_grid, p_bad_eff)  # 执行本行逻辑
        if np.isnan(confN):  # 条件判断
            continue  # 执行本行逻辑
        if confN >= conf_target:  # 条件判断
            return N  # 返回结果
    return None  # 返回结果

def prior_pass_prob(N: int, prior_dens: np.ndarray, R_grid: np.ndarray) -> float:  # 定义函数 prior_pass_prob
    if N is None or N <= 0:  # 条件判断
        return np.nan  # 返回结果

    delta = R_grid[1] - R_grid[0]  # 计算网格步长
    out = (np.power(R_grid, N) * prior_dens).sum() * delta  # 构造摘要输出字典
    return float(np.clip(out, 0.0, 1.0))  # 返回结果

def p_prior_implied(N: int, pass_prob: float) -> float:  # 定义函数 p_prior_implied
    if N is None or N <= 0 or np.isnan(pass_prob):  # 条件判断
        return np.nan  # 返回结果

    pp = float(np.clip(pass_prob, 0.0, 1.0))  # 执行本行逻辑
    return 1.0 - (pp ** (1.0 / N))  # 返回结果



def posterior_density_zero_fail(N: int, prior_dens: np.ndarray, R_grid: np.ndarray) -> np.ndarray:  # 定义函数 posterior_density_zero_fail
    delta = R_grid[1] - R_grid[0]  # 计算网格步长

    if N is None or N <= 0:  # 条件判断
        dens = np.ones_like(R_grid, dtype=float)  # 构造占位密度
        return dens / (dens.sum() * delta)  # 返回归一化占位密度

    like = np.power(R_grid, N)  # 计算零失效条件下的似然项 R^N
    post = np.maximum(prior_dens * like, 0.0)  # 计算未归一化后验
    Z = post.sum() * delta  # 计算归一化常数

    if Z <= 0:  # 条件判断
        dens = np.ones_like(R_grid, dtype=float)  # 构造占位密度
        return dens / (dens.sum() * delta)  # 返回归一化占位密度

    return post / Z  # 返回零失效条件下的后验密度


def posterior_mean_R_from_density(post_dens: np.ndarray, R_grid: np.ndarray) -> float:  # 定义函数 posterior_mean_R_from_density
    delta = R_grid[1] - R_grid[0]  # 计算网格步长
    return float(np.clip((R_grid * post_dens).sum() * delta, 0.0, 1.0))  # 返回后验平均可靠率


def sample_R_from_density(seed: int, post_dens: np.ndarray, R_grid: np.ndarray) -> float:  # 定义函数 sample_R_from_density
    rng = np.random.default_rng(seed)  # 初始化随机数生成器
    delta = R_grid[1] - R_grid[0]  # 计算网格步长
    prob = np.maximum(post_dens.astype(float), 0.0) * delta  # 将密度离散化为概率质量

    if prob.sum() <= 0:  # 条件判断
        prob = np.ones_like(prob) / prob.size  # 回退为均匀分布
    else:  # 进入另一分支
        prob = prob / prob.sum()  # 归一化离散概率

    idx = int(rng.choice(prob.size, p=prob))  # 按后验分布抽取一个网格位置
    return float(np.clip(R_grid[idx], 0.0, 1.0))  # 返回抽到的可靠率


def simulate_2025_pool_from_posterior(seed: int, n_batches: int, per_batch: int, post_dens: np.ndarray, R_grid: np.ndarray):  # 定义函数 simulate_2025_pool_from_posterior
    R_draw = sample_R_from_density(seed, post_dens, R_grid)  # 从后验分布抽取一个可靠率
    p_draw = float(np.clip(1.0 - R_draw, 0.0, 1.0))  # 换算为对应失效率
    df_pool = simulate_2025_pool(seed=seed + 123, n_batches=n_batches, per_batch=per_batch, p_true=p_draw)  # 用后验抽到的失效率生成新的测试池
    return df_pool, R_draw, p_draw  # 返回新的测试池以及抽到的 R / p

def pick_writable_dir() -> str:  # 定义函数 pick_writable_dir
    cand1 = os.path.join(os.path.expanduser("~"), "output_A")  # 设置首选输出目录

    try:  # 尝试执行
        os.makedirs(cand1, exist_ok=True)  # 执行本行逻辑
        testfile = os.path.join(cand1, ".write_test")  # 执行本行逻辑
        with open(testfile, "w") as f:  # 打开文件并写入
            f.write("ok")  # 执行本行逻辑
        os.remove(testfile)  # 执行本行逻辑
        return cand1  # 返回结果
    except Exception:  # 异常处理
        cand2 = os.path.join(tempfile.gettempdir(), "output_A")  # 设置备用输出目录
        os.makedirs(cand2, exist_ok=True)  # 执行本行逻辑
        return cand2  # 返回结果


# =========================================================
# Simulation helpers
# =========================================================

def build_fixed_ppm_table() -> pd.DataFrame:  # 定义函数 build_fixed_ppm_table
    """  # 执行本行逻辑
    强波动版本：  # 执行本行逻辑
    - 保留年内震荡  # 执行本行逻辑
    - 不强制全局严格递减  # 执行本行逻辑
    - 只要求整体趋势由 HIST_CLASS_DPPM 控制为下降  # 执行本行逻辑

 \\\
    1) 历史起始 2020 seg1 调整为 13000 ppm  # 执行本行逻辑
    2) 历史结尾 2024 seg5 调整为 140 ppm  # 执行本行逻辑
    其余位置保持原强波动设定不变  # 执行本行逻辑
    """  # 执行本行逻辑
    rows = []  # 初始化结果列表

    for yi, cls in enumerate(HIST_CLASS_NAMES):  # 开始循环
        base_ppm = HIST_CLASS_DPPM[yi]  # 执行本行逻辑
        ppm_year = np.round(base_ppm * SEG_FACTOR).astype(int)  # 执行本行逻辑
        ppm_year = np.maximum(ppm_year, 1)  # 执行本行逻辑

        for s in range(1, SEG_K + 1):  # 开始循环
            rows.append({  # 追加到列表中
                "类别": cls,  # 执行本行逻辑
                "段": s,  # 执行本行逻辑
                "X_ppm": int(ppm_year[s - 1])  # 执行本行逻辑
            })

    df = pd.DataFrame(rows)  # 构造数据表
    df = df.sort_values(["类别", "段"]).reset_index(drop=True)  # 构造数据表

    # 仅改两处以与弱波动场景首尾匹配
    df.loc[(df["类别"] == "2020") & (df["段"] == 1), "X_ppm"] = 13000  # 按位置或条件取值
    df.loc[(df["类别"] == "2024") & (df["段"] == 5), "X_ppm"] = 140  # 按位置或条件取值

    return df  # 返回结果

def simulate_hist_2020_2024(seed: int, ppm_table: pd.DataFrame,  # 定义函数 simulate_hist_2020_2024
                            per_batch: int = 4000, n_per_seg_batch: int = 1) -> pd.DataFrame:  # 执行本行逻辑
    rng = np.random.default_rng(seed)  # 初始化随机数生成器
    rows = []  # 初始化结果列表

    for cls in HIST_CLASS_NAMES:  # 开始循环
        for s in range(1, SEG_K + 1):  # 开始循环
            seg_ppm = int(  # 读取当前分段 ppm
                ppm_table.loc[  # 按位置或条件取值
                    (ppm_table["类别"] == cls) & (ppm_table["段"] == s),  # 执行本行逻辑
                    "X_ppm"  # 执行本行逻辑
                ].iloc[0]  # 按位置或条件取值
            )

            # 保持你原来模拟口径
            p_fail = seg_ppm / (10 * 1e6)  # 换算当前分段失效率

            for b in range(1, n_per_seg_batch + 1):  # 开始循环
                fail_vec = rng.binomial(1, p_fail, size=per_batch)  # 生成失效指示向量
                rows.append(pd.DataFrame({  # 追加到列表中
                    "批次编号": [f"{cls}_S{s}_L{b}"] * per_batch,  # 执行本行逻辑
                    "类别": [cls] * per_batch,  # 执行本行逻辑
                    "段": [s] * per_batch,  # 执行本行逻辑
                    "X_ppm": [seg_ppm] * per_batch,  # 执行本行逻辑
                    "是否失效": fail_vec.astype(int)  # 执行本行逻辑
                }))  # 执行本行逻辑

    return pd.concat(rows, ignore_index=True)  # 返回结果

def simulate_2025_pool(seed: int, n_batches: int, per_batch: int, p_true: float) -> pd.DataFrame:  # 定义函数 simulate_2025_pool
    rng = np.random.default_rng(seed)  # 初始化随机数生成器
    total = n_batches * per_batch  # 计算总样本量
    fail = rng.binomial(1, p_true, size=total).astype(int)  # 生成 2025 失效指示
    batch_id = np.repeat(np.arange(1, n_batches + 1), per_batch)  # 生成批次编号

    return pd.DataFrame({"批次编号": batch_id, "类别": "2025", "是否失效": fail})  # 返回结果

def summarise_batches(df_hist: pd.DataFrame) -> pd.DataFrame:  # 定义函数 summarise_batches
    g = df_hist.groupby(["批次编号", "类别", "段"], as_index=False).agg(  # 按批次做汇总统计
        n=("是否失效", "size"),  # 执行本行逻辑
        r=("是否失效", "sum"),  # 执行本行逻辑
        X_ppm=("X_ppm", "first")  # 执行本行逻辑
    )

    g["R_hat"] = np.clip(1.0 - g["r"] / g["n"], 0.0, 1.0)  # 计算经验可靠率
    g["p_hat_batch"] = np.clip(g["r"] / g["n"], 0.0, 1.0)  # 计算经验失效率
    g["p_true_local"] = g["X_ppm"] / 1e6  # 换算局部真实失效率
    return g  # 返回结果


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

def prior_hist_given_x(R_grid: np.ndarray, batch_tbl: pd.DataFrame,  # 定义函数 prior_hist_given_x
                       x_ppm: int, target_time_index: float,  # 执行本行逻辑
                       hX: float, hT: float, hR: float) -> np.ndarray:  # 执行本行逻辑
    w_ppm = norm.pdf((x_ppm - batch_tbl["X_ppm"].to_numpy(dtype=float)) / hX)  # 计算 ppm 相似性权重

    hist_time_index = np.array([  # 构造历史时间索引数组
        make_time_index(y, s, base_year=BASE_YEAR, seg_k=SEG_K)  # 执行本行逻辑
        for y, s in zip(batch_tbl["类别"].astype(str), batch_tbl["段"])  # 开始循环
    ], dtype=float)  # 执行本行逻辑

    w_time = norm.pdf((target_time_index - hist_time_index) / hT)  # 计算时间相似性权重
    w = w_ppm * w_time  # 构造权重

    return kde1d_density_weighted(  # 返回结果
        R_grid,  # 执行本行逻辑
        batch_tbl["R_hat"].to_numpy(dtype=float),  # 执行本行逻辑
        w,  # 执行本行逻辑
        hR  # 执行本行逻辑
    )

def prior_new_from_proxy(R_grid: np.ndarray, batch_tbl: pd.DataFrame, hR: float) -> np.ndarray:  # 定义函数 prior_new_from_proxy
    tbl_new = batch_tbl[  # 筛选代理批次数据
        (batch_tbl["类别"].astype(str) == str(NEW_PROXY_CLASS)) &  # 执行本行逻辑
        (batch_tbl["段"] == NEW_PROXY_SEG)  # 执行本行逻辑
    ]

    if tbl_new.shape[0] == 0:  # 条件判断
        delta = R_grid[1] - R_grid[0]  # 计算网格步长
        dens = np.ones_like(R_grid, dtype=float)  # 构造密度数组
        return dens / (dens.sum() * delta)  # 返回结果

    return kde1d_density(R_grid, tbl_new["R_hat"].to_numpy(dtype=float), hR)  # 返回结果

def mix_prior(prior_hist_x: np.ndarray, prior_new: np.ndarray, R_grid: np.ndarray, alpha: float) -> np.ndarray:  # 定义函数 mix_prior
    delta = R_grid[1] - R_grid[0]  # 计算网格步长
    prior = alpha * prior_hist_x + (1.0 - alpha) * prior_new  # 混合两部分先验
    prior = np.maximum(prior, 0.0)  # 混合两部分先验
    prior = prior / (prior.sum() * delta)  # 混合两部分先验
    return prior  # 返回结果

def estimate_qhat_weighted_split_fixed(batch_tbl_fixed: pd.DataFrame,  # 定义函数 estimate_qhat_weighted_split_fixed
                                       build_ids: np.ndarray, calib_ids: np.ndarray,  # 执行本行逻辑
                                       alpha_now: float, R_grid: np.ndarray,  # 执行本行逻辑
                                       hX: float, hT: float, hR: float, conf_level: float,  # 执行本行逻辑
                                       target_time_index: float) -> float:  # 执行本行逻辑
    build = batch_tbl_fixed.iloc[build_ids].copy()  # 取 build 数据集
    calib = batch_tbl_fixed.iloc[calib_ids].copy()  # 取 calibration 数据集

    if calib.shape[0] == 0:  # 条件判断
        return 0.0  # 返回结果

    prior_new_build = prior_new_from_proxy(R_grid, build, hR)  # 构造 build 集上的最近先验
    delta = R_grid[1] - R_grid[0]  # 计算网格步长

    phat = np.zeros(calib.shape[0], dtype=float)  # 初始化预测失效率数组
    ptrue = calib["p_true_local"].to_numpy(dtype=float)  # 读取校准集真实失效率代理值

    for i in range(calib.shape[0]):  # 开始循环
        x_i = int(calib["X_ppm"].iloc[i])  # 读取当前样本 ppm

        prior_hist_xi = prior_hist_given_x(  # 构造当前样本历史先验
            R_grid=R_grid,  # 执行本行逻辑
            batch_tbl=build,  # 执行本行逻辑
            x_ppm=x_i,  # 执行本行逻辑
            target_time_index=target_time_index,  # 执行本行逻辑
            hX=hX, hT=hT, hR=hR  # 执行本行逻辑
        )

        prior_xi = mix_prior(prior_hist_xi, prior_new_build, R_grid, alpha=alpha_now)  # 混合当前样本先验
        ER = (R_grid * prior_xi).sum() * delta  # 计算先验下的 E[R]
        phat[i] = 1.0 - ER  # 执行本行逻辑

    resid = ptrue - phat  # 计算残差

    # 不截断 qhat
    qhat = float(np.quantile(resid, conf_level, method="linear"))  # 计算残差分位数 qhat
    return qhat  # 返回结果


# =========================================================
# Core runner for one (ppm2025, LTPD_ppm)
# =========================================================

def run_one_setting(batch_tbl_fixed: pd.DataFrame,  # 定义函数 run_one_setting
                    ppm_2025: int,  # 执行本行逻辑
                    ltpd_ppm: int,  # 执行本行逻辑
                    safe_out_dir: str = None,  # 执行本行逻辑
                    save_rep: bool = False):  # 执行本行逻辑

    R_grid = np.arange(0.0, 1.0001, 0.001)  # 构造可靠率网格

    proxy_rows = batch_tbl_fixed[  # 筛选代理批次行
        (batch_tbl_fixed["类别"].astype(str) == str(NEW_PROXY_CLASS)) &  # 执行本行逻辑
        (batch_tbl_fixed["段"] == NEW_PROXY_SEG)  # 执行本行逻辑
    ]

    if proxy_rows.shape[0] == 0:  # 条件判断
        raise RuntimeError(  # 抛出运行错误
            f"Cannot find proxy rows for NEW_PROXY_CLASS={NEW_PROXY_CLASS}, NEW_PROXY_SEG={NEW_PROXY_SEG}."  # 执行本行逻辑
        )

    x_new_ppm = int(proxy_rows["X_ppm"].iloc[0])  # 读取代理输入 ppm
    if XNEW_TIMES_10:  # 条件判断
        x_new_ppm = int(10 * x_new_ppm)  # 读取代理输入 ppm

    TARGET_TIME_INDEX = len(HIST_CLASS_NAMES) * SEG_K  # 设置目标时间索引

    prior_hist_xnew_fixed = prior_hist_given_x(  # 构造固定历史先验
        R_grid, batch_tbl_fixed,  # 执行本行逻辑
        x_ppm=x_new_ppm,  # 执行本行逻辑
        target_time_index=TARGET_TIME_INDEX,  # 执行本行逻辑
        hX=HX_PPM, hT=HT_TIME, hR=hR  # 执行本行逻辑
    )

    prior_new_fixed = prior_new_from_proxy(R_grid, batch_tbl_fixed, hR=hR)  # 构造固定最近先验

    rng_split = np.random.default_rng(SPLIT_SEED_FIXED)  # 初始化固定划分随机数
    id_all = rng_split.permutation(batch_tbl_fixed.shape[0])  # 随机打乱批次索引

    n_build = max(1, int(np.floor(BUILD_FRAC * batch_tbl_fixed.shape[0])))  # 计算 build 集大小
    build_ids = id_all[:n_build]  # 保存 build 索引
    calib_ids = id_all[n_build:]  # 保存 calibration 索引

    if calib_ids.size == 0:  # 条件判断
        raise RuntimeError("Calibration split empty. Reduce BUILD_FRAC or increase data.")  # 抛出运行错误

    rep_rows = []  # 初始化 replication 结果列表
    alpha_t = alpha_init  # 初始化在线权重 alpha
    err_hist = []  # 初始化误差历史

    NEW_p_true_2025 = ppm_2025 / 1e6  # 换算 2025 真实失效率
    P_BAD = ltpd_ppm / 1e6  # 换算当前 LTPD 为概率

    x_new_ppm_base = int(x_new_ppm / 10) if XNEW_TIMES_10 else int(x_new_ppm)  # 还原代理 ppm 基础口径
    p_prior_fixed_from_xnew = (x_new_ppm_base / 1e6)  # 计算基准先验失效率

    ERR_SCALE_LOCAL = max(NEW_p_true_2025, 1e-6)  # 设置局部误差尺度

    for t in range(1, N_REP + 1):  # 开始循环
        seed_2025 = SEED_BASE + 999999 + t  # 设置当前轮 2025 随机种子
        alpha_before = alpha_t  # 记录更新前 alpha

        prior_dens_xnew = mix_prior(prior_hist_xnew_fixed, prior_new_fixed, R_grid, alpha=alpha_before)  # 构造当前轮先验密度

        qhat_raw = estimate_qhat_weighted_split_fixed(  # 估计原始 qhat
            batch_tbl_fixed=batch_tbl_fixed,  # 执行本行逻辑
            build_ids=build_ids,  # 执行本行逻辑
            calib_ids=calib_ids,  # 执行本行逻辑
            alpha_now=alpha_before,  # 执行本行逻辑
            R_grid=R_grid,  # 执行本行逻辑
            hX=HX_PPM, hT=HT_TIME, hR=hR,  # 执行本行逻辑
            conf_level=conf_level,  # 执行本行逻辑
            target_time_index=TARGET_TIME_INDEX  # 执行本行逻辑
        )

        qhat_use = qhat_raw  # 设置当前使用的 qhat

        N_star = solve_Nstar_by_postconf(  # 反解最小样本量
            prior_dens=prior_dens_xnew,  # 执行本行逻辑
            R_grid=R_grid,  # 执行本行逻辑
            qhat=qhat_use,  # 执行本行逻辑
            p_bad=P_BAD,  # 执行本行逻辑
            conf_target=CONF_TARGET,  # 执行本行逻辑
            N_max=N_max  # 执行本行逻辑
        )

        p_bad_eff_now = clamp01(P_BAD - qhat_use)  # 计算当前 replication 下经 qhat 修正后的有效坏品率门槛
        posterior_conf_at_nstar = posterior_conf_good(N_star, prior_dens_xnew, R_grid, p_bad_eff_now) if N_star is not None else np.nan  # 计算 N* 对应的后验好批次置信度
        post_dens_zero_fail = posterior_density_zero_fail(N_star, prior_dens_xnew, R_grid) if N_star is not None else np.nan  # 构造 N* 对应零失效条件下的后验分布
        posterior_mean_R = posterior_mean_R_from_density(post_dens_zero_fail, R_grid) if N_star is not None else np.nan  # 计算后验平均可靠率
        posterior_mean_p = float(np.clip(1.0 - posterior_mean_R, 0.0, 1.0)) if N_star is not None and not np.isnan(posterior_mean_R) else np.nan  # 计算后验平均失效率

        df_2025_pool = simulate_2025_pool(  # 生成 2025 测试池
            seed=seed_2025,  # 执行本行逻辑
            n_batches=Y25_N_BATCHES_POOL,  # 执行本行逻辑
            per_batch=Y25_PER_BATCH_POOL,  # 执行本行逻辑
            p_true=NEW_p_true_2025  # 执行本行逻辑
        )

        # 新增：再用当前 replication 下的后验分布生成一个新的 2025 测试池，用来验证同一个 N* 在“后验生成池子”中的通过率表现
        if N_star is not None:  # 条件判断
            df_2025_pool_post, R_post_draw, p_post_draw = simulate_2025_pool_from_posterior(
                seed=seed_2025 + 333333,  # 执行本行逻辑
                n_batches=Y25_N_BATCHES_POOL,  # 执行本行逻辑
                per_batch=Y25_PER_BATCH_POOL,  # 执行本行逻辑
                post_dens=post_dens_zero_fail,  # 执行本行逻辑
                R_grid=R_grid  # 执行本行逻辑
            )  # 用后验分布生成新的测试池
        else:  # 进入另一分支
            df_2025_pool_post, R_post_draw, p_post_draw = None, np.nan, np.nan  # 若 N* 无效，则后验生成池子置空

        test_n = df_2025_pool.shape[0]  # 读取测试池总样本量
        replace_flag = False  # 初始化是否有放回抽样标记

        if N_star is not None and N_star > test_n:  # 条件判断
            if TEST_SAMPLE_REPLACE_IF_NEEDED:  # 条件判断
                replace_flag = True  # 初始化是否有放回抽样标记
            else:  # 进入另一分支
                N_star = None  # 反解最小样本量

        F_t = None  # 初始化当前轮失效数
        p_obs_t = np.nan  # 初始化当前轮观测失效率
        passed = None  # 初始化当前轮是否通过
        F_t_post = None  # 新增：初始化后验生成池子的失效数
        p_obs_t_post = np.nan  # 新增：初始化后验生成池子的观测失效率
        passed_post = None  # 新增：初始化后验生成池子的是否通过

        if N_star is not None and N_star > 0:  # 条件判断
            rng = np.random.default_rng(seed_2025 + 777)  # 初始化随机数生成器
            idx = rng.choice(test_n, size=N_star, replace=replace_flag)  # 从真实测试池中抽取 N* 个样本
            F_t = int(df_2025_pool["是否失效"].to_numpy()[idx].sum())  # 统计真实测试池中的失效数
            p_obs_t = F_t / N_star  # 计算真实测试池中的观测失效率
            passed = int(pass_rule_test(F_t))  # 判断真实测试池是否通过

            test_n_post = df_2025_pool_post.shape[0]  # 读取后验生成池子的总样本量
            replace_flag_post = bool(TEST_SAMPLE_REPLACE_IF_NEEDED and N_star > test_n_post)  # 设置后验生成池子的抽样方式
            rng_post = np.random.default_rng(seed_2025 + 888)  # 初始化后验生成池子的抽样随机数生成器
            idx_post = rng_post.choice(test_n_post, size=N_star, replace=replace_flag_post)  # 从后验生成池子中抽取 N* 个样本
            F_t_post = int(df_2025_pool_post["是否失效"].to_numpy()[idx_post].sum())  # 统计后验生成池子中的失效数
            p_obs_t_post = F_t_post / N_star  # 计算后验生成池子的观测失效率
            passed_post = int(pass_rule_test(F_t_post))  # 判断后验生成池子是否通过

        pass_prob = prior_pass_prob(N_star, prior_dens_xnew, R_grid) if N_star is not None else np.nan  # 计算先验通过概率
        p_prior_imp = p_prior_implied(N_star, pass_prob) if N_star is not None else np.nan  # 反推等效先验失效率

        p_prior_t = p_prior_fixed_from_xnew * 10.0  # 设置先验比较基准
        err_t = (p_obs_t - p_prior_t) if not np.isnan(p_obs_t) else np.nan  # 计算当前轮误差

        if not np.isnan(err_t):  # 条件判断
            err_hist.append(err_t)  # 追加到列表中

        err_smooth = np.nan  # 初始化平滑误差
        if len(err_hist) > 0:  # 条件判断
            D = min(WIN, len(err_hist))  # 确定当前窗口长度
            err_smooth = float(np.mean(err_hist[-D:]))  # 初始化平滑误差

            delta_alpha = clamp_step(STEP_SIZE * (err_smooth / ERR_SCALE_LOCAL))  # 计算 alpha 更新量
            alpha_t = clamp_alpha(alpha_t - delta_alpha)  # 初始化在线权重 alpha

        alpha_after = alpha_t  # 记录更新后 alpha

        rep_rows.append({  # 追加到列表中
            "rep_id": t,  # 执行本行逻辑
            "ppm_2025": ppm_2025,  # 执行本行逻辑
            "ltpd_ppm": ltpd_ppm,  # 执行本行逻辑
            "alpha_before": alpha_before,  # 执行本行逻辑
            "alpha_after": alpha_after,  # 执行本行逻辑
            "N_star": N_star,  # 执行本行逻辑
            "F_t": F_t,  # 执行本行逻辑
            "p_obs_t": p_obs_t,  # 执行本行逻辑
            "p_prior_imp": p_prior_imp,  # 执行本行逻辑
            "p_prior_xnew": p_prior_t,  # 执行本行逻辑
            "posterior_conf_at_nstar": posterior_conf_at_nstar,  # 新增：记录 N* 对应的后验置信度
            "posterior_mean_R": posterior_mean_R,  # 新增：记录后验平均可靠率
            "posterior_mean_p": posterior_mean_p,  # 新增：记录后验平均失效率
            "posterior_draw_R": R_post_draw,  # 新增：记录后验生成池子时抽到的可靠率
            "posterior_draw_p": p_post_draw,  # 新增：记录后验生成池子时抽到的失效率
            "F_t_post": F_t_post,  # 新增：记录后验生成池子的失效数
            "p_obs_t_post": p_obs_t_post,  # 新增：记录后验生成池子的观测失效率
            "pass_post": passed_post,  # 新增：记录后验生成池子的通过结果
            "err_t": err_t,  # 执行本行逻辑
            "err_smooth": err_smooth,  # 执行本行逻辑
            "pass": passed,  # 执行本行逻辑
            "qhat_raw": qhat_raw,  # 执行本行逻辑
            "qhat_used": qhat_use  # 执行本行逻辑
        })

    rep_df = pd.DataFrame(rep_rows)  # 整理 replication 结果表

    if save_rep and safe_out_dir is not None:  # 条件判断
        fn_rep = os.path.join(  # 构造逐轮结果文件名
            safe_out_dir,  # 执行本行逻辑
            f"rep_strong_ppm2025_{ppm_2025}_ltpd_{ltpd_ppm}.csv"  # 执行本行逻辑
        )
        rep_df.to_csv(fn_rep, index=False)  # 导出结果文件

    pass_rate = float(rep_df["pass"].dropna().mean()) if rep_df["pass"].notna().any() else np.nan  # 计算真实测试池通过率
    pass_rate_post = float(rep_df["pass_post"].dropna().mean()) if rep_df["pass_post"].notna().any() else np.nan  # 计算后验生成测试池通过率
    posterior_conf_mean = float(rep_df["posterior_conf_at_nstar"].dropna().mean()) if rep_df["posterior_conf_at_nstar"].notna().any() else np.nan  # 计算平均后验置信度

    out = {  # 构造摘要输出字典
        "2025_ppm": ppm_2025,  # 执行本行逻辑
        "LTPD_ppm": ltpd_ppm,  # 执行本行逻辑
        "N1_star": int(rep_df.loc[rep_df["rep_id"] == 1, "N_star"].iloc[0]) if rep_df["N_star"].notna().any() else np.nan,  # 按位置或条件取值
        "N100_star": int(rep_df.loc[rep_df["rep_id"] == N_REP, "N_star"].iloc[0]) if rep_df["N_star"].notna().any() else np.nan,  # 按位置或条件取值
        "Nstar_mean": float(rep_df["N_star"].dropna().mean()) if rep_df["N_star"].notna().any() else np.nan,  # 计算平均值
        "Nstar_min": int(rep_df["N_star"].dropna().min()) if rep_df["N_star"].notna().any() else np.nan,  # 计算最值
        "Nstar_max": int(rep_df["N_star"].dropna().max()) if rep_df["N_star"].notna().any() else np.nan,  # 计算最值
        "posterior_conf_mean": posterior_conf_mean,  # 新增：记录 N* 对应后验置信度的平均值
        "pass_rate": pass_rate,  # 执行本行逻辑
        "post_pool_pass_rate": pass_rate_post,  # 新增：记录后验生成测试池通过率
        "producer_risk": 1.0 - pass_rate if pd.notna(pass_rate) else np.nan,  # 执行本行逻辑
        "consumer_risk": pass_rate if pd.notna(pass_rate) else np.nan,  # 执行本行逻辑
        "producer_risk_post": 1.0 - pass_rate_post if pd.notna(pass_rate_post) else np.nan,  # 新增：后验生成池子的生产者风险
        "consumer_risk_post": pass_rate_post if pd.notna(pass_rate_post) else np.nan  # 新增：后验生成池子的消费者风险
    }

    return out, rep_df  # 返回结果


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

def main():  # 定义函数 main
    safe_out_dir = pick_writable_dir()  # 选择输出目录
    print(f"\nFiles will be saved to: {safe_out_dir}\n")  # 打印提示信息

    # Step 1: 先构造强波动历史数据，并按公司要求的输入格式导出 Excel / CSV
    ppm_table_fixed_sim = build_fixed_ppm_table()  # 构造强波动 ppm 场景表

    seed_hist_fixed_sim = SEED_BASE + 1  # 设置历史模拟种子
    df_hist_fixed_sim = simulate_hist_2020_2024(  # 生成历史样本级数据
        seed_hist_fixed_sim,  # 执行本行逻辑
        ppm_table_fixed_sim,  # 执行本行逻辑
        HIST_PER_SEG_BATCH,  # 执行本行逻辑
        N_PER_SEG_BATCH  # 执行本行逻辑
    )
    batch_tbl_fixed_sim = summarise_batches(df_hist_fixed_sim)  # 汇总历史批次表（模拟版）

    if GENERATE_SIM_COMPANY_INPUT:  # 若需要先导出公司格式输入文件
        export_simulation_to_company_format(batch_tbl_fixed_sim, REAL_INPUT_PATH)  # 导出公司输入格式 Excel / CSV

    if USE_REAL_INPUT:  # 若主程序后续优先从公司格式文件读取
        batch_tbl_fixed = load_real_batch_table(REAL_INPUT_PATH, REAL_INPUT_SHEET)  # 按公司格式重新读回历史批次表
    else:  # 否则仍直接使用模拟汇总表
        batch_tbl_fixed = batch_tbl_fixed_sim  # 保持原有内部模拟逻辑

    # 保存历史输入表
    hist_input_path = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_historical_input.csv")  # 构造历史输入表路径
    batch_tbl_fixed.to_csv(hist_input_path, index=False, encoding="utf-8-sig")  # 导出结果文件

    # 也把强波动 ppm 场景表保存出来，方便检查
    scenario_path = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_ppm_scenario_table.csv")  # 构造场景表路径
    ppm_table_fixed_sim.to_csv(scenario_path, index=False, encoding="utf-8-sig")  # 导出结果文件

    # Step 2: 跑所有组合
    all_summary = []  # 初始化总摘要列表

    total_jobs = len(PPM_2025_GRID_ALL) * len(LTPD_PPM_GRID)  # 计算总任务数
    job_id = 0  # 初始化任务计数器

    for ppm_2025 in PPM_2025_GRID_ALL:  # 开始循环
        for ltpd_ppm in LTPD_PPM_GRID:  # 开始循环
            job_id += 1  # 执行本行逻辑
            print(f"[{job_id}/{total_jobs}] Running strong scenario: ppm_2025={ppm_2025}, LTPD_ppm={ltpd_ppm} ...")  # 打印提示信息

            summary_row, _ = run_one_setting(  # 运行单个组合并获取摘要
                batch_tbl_fixed=batch_tbl_fixed,  # 执行本行逻辑
                ppm_2025=ppm_2025,  # 执行本行逻辑
                ltpd_ppm=ltpd_ppm,  # 执行本行逻辑
                safe_out_dir=safe_out_dir,  # 执行本行逻辑
                save_rep=False  # 执行本行逻辑
            )
            all_summary.append(summary_row)  # 追加到列表中

    summary_df = pd.DataFrame(all_summary)  # 整理全部摘要结果

    # 保存完整长表
    fn_summary = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_summary_long.csv")  # 构造完整长表路径
    summary_df.to_csv(fn_summary, index=False, encoding="utf-8-sig")  # 导出结果文件

    # N1 宽表
    n1_wide = summary_df.pivot(index="2025_ppm", columns="LTPD_ppm", values="N1_star").reset_index()  # 生成 N1 宽表
    fn_n1 = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_N1_wide.csv")  # 构造 N1 宽表路径
    n1_wide.to_csv(fn_n1, index=False, encoding="utf-8-sig")  # 导出结果文件

    # N100 宽表
    n100_wide = summary_df.pivot(index="2025_ppm", columns="LTPD_ppm", values="N100_star").reset_index()  # 生成 N100 宽表
    fn_n100 = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_N100_wide.csv")  # 构造 N100 宽表路径
    n100_wide.to_csv(fn_n100, index=False, encoding="utf-8-sig")  # 导出结果文件

    # 通过率宽表
    pass_wide = summary_df.pivot(index="2025_ppm", columns="LTPD_ppm", values="pass_rate").reset_index()  # 生成真实测试池通过率宽表
    fn_pass = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_passrate_wide.csv")  # 构造通过率宽表路径
    pass_wide.to_csv(fn_pass, index=False, encoding="utf-8-sig")  # 导出结果文件

    # Step 5b: 新增“纯 posterior validation”摘要表
    posterior_only_rows = []  # 用于保存纯 posterior validation 摘要结果
    for ltpd_ppm in LTPD_PPM_GRID:  # 逐个 LTPD 运行 posterior-only 验证
        print(f"[posterior-only] Running strong scenario: LTPD_ppm={ltpd_ppm} ...")  # 打印 posterior-only 运行信息
        posterior_row, _ = run_posterior_only_setting(  # 固定 x_new 和历史数据，仅用后验生成池子来验证 N*
            batch_tbl_fixed=batch_tbl_fixed,  # 传入固定历史批次摘要表
            ltpd_ppm=ltpd_ppm,  # 当前 LTPD
            safe_out_dir=safe_out_dir,  # 输出目录
            save_rep=False  # 不单独保存 replication 级结果
        )
        posterior_only_rows.append(posterior_row)  # 保存 posterior-only 摘要结果

    posterior_only_df = pd.DataFrame(posterior_only_rows)  # 整理纯 posterior validation 摘要表
    posterior_only_df = posterior_only_df[["LTPD_ppm", "N1_star", "N100_star", "posterior_conf_at_nstar", "posterior_pool_pass_rate"]].copy()  # 按公司要求保留固定列顺序
    fn_posterior_only = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_posterior_only_validation.csv")  # 构造纯 posterior validation 摘要表路径
    posterior_only_df.to_csv(fn_posterior_only, index=False, encoding="utf-8-sig")  # 导出纯 posterior validation 摘要表


    # 新增：后验生成测试池的通过率宽表
    pass_post_wide = summary_df.pivot(index="2025_ppm", columns="LTPD_ppm", values="post_pool_pass_rate").reset_index()  # 生成后验生成测试池通过率宽表
    fn_pass_post = os.path.join(safe_out_dir, f"{OUTPUT_BASENAME}_posterior_pool_passrate_wide.csv")  # 构造后验生成测试池通过率宽表路径
    pass_post_wide.to_csv(fn_pass_post, index=False, encoding="utf-8-sig")  # 导出结果文件

    # 针对某一个固定 LTPD，输出公司风格摘要表
    selected_df = summary_df[summary_df["LTPD_ppm"] == SELECTED_LTPD_PPM_FOR_SUMMARY].copy()  # 筛选固定 LTPD 的结果

    good_df = selected_df[selected_df["2025_ppm"].isin(PPM_2025_GRID_GOOD)].copy()  # 提取好批次结果
    good_df = good_df.sort_values("2025_ppm").reset_index(drop=True)  # 提取好批次结果
    good_df["好批次通过率=通过率"] = good_df["pass_rate"]  # 添加真实测试池下的好批次通过率列
    good_df["后验生成池通过率"] = good_df["post_pool_pass_rate"]  # 新增：添加后验生成测试池下的好批次通过率列
    good_df["生产者风险=1-好批次通过率"] = 1.0 - good_df["pass_rate"]  # 添加真实测试池下的生产者风险列

    good_out = good_df[[  # 整理好批次摘要表
        "2025_ppm", "N1_star", "N100_star",  # 执行本行逻辑
        "好批次通过率=通过率", "后验生成池通过率", "生产者风险=1-好批次通过率"  # 执行本行逻辑
    ]].copy()  # 复制数据避免原地修改
    good_out = good_out.rename(columns={"2025_ppm": "2025 年批次产片的全生命周期ppm（好批次）"})  # 整理好批次摘要表
    fn_good = os.path.join(  # 构造好批次摘要表路径
        safe_out_dir,  # 执行本行逻辑
        f"{OUTPUT_BASENAME}_good_batches_LTPD_{SELECTED_LTPD_PPM_FOR_SUMMARY}.csv"  # 执行本行逻辑
    )
    good_out.to_csv(fn_good, index=False, encoding="utf-8-sig")  # 导出结果文件

    bad_df = selected_df[selected_df["2025_ppm"].isin(PPM_2025_GRID_BAD)].copy()  # 提取坏批次结果
    bad_df = bad_df.sort_values("2025_ppm").reset_index(drop=True)  # 提取坏批次结果
    bad_df["坏批次通过率=通过率"] = bad_df["pass_rate"]  # 添加真实测试池下的坏批次通过率列
    bad_df["后验生成池通过率"] = bad_df["post_pool_pass_rate"]  # 新增：添加后验生成测试池下的坏批次通过率列
    bad_df["消费者风险=坏批次通过率"] = bad_df["pass_rate"]  # 添加真实测试池下的消费者风险列

    bad_out = bad_df[[  # 整理坏批次摘要表
        "2025_ppm", "N1_star", "N100_star",  # 执行本行逻辑
        "坏批次通过率=通过率", "后验生成池通过率", "消费者风险=坏批次通过率"  # 执行本行逻辑
    ]].copy()  # 复制数据避免原地修改
    bad_out = bad_out.rename(columns={"2025_ppm": "2025 年批次产片的全生命周期ppm（坏批次）"})  # 整理坏批次摘要表
    fn_bad = os.path.join(  # 构造坏批次摘要表路径
        safe_out_dir,  # 执行本行逻辑
        f"{OUTPUT_BASENAME}_bad_batches_LTPD_{SELECTED_LTPD_PPM_FOR_SUMMARY}.csv"  # 执行本行逻辑
    )
    bad_out.to_csv(fn_bad, index=False, encoding="utf-8-sig")  # 导出结果文件

    print("\n===== DONE =====")  # 打印提示信息
    print("历史输入表:", hist_input_path)  # 打印提示信息
    print("强波动 ppm 场景表:", scenario_path)  # 打印提示信息
    print("完整长表:", fn_summary)  # 打印提示信息
    print("N1 宽表:", fn_n1)  # 打印提示信息
    print("N100 宽表:", fn_n100)  # 打印提示信息
    print("通过率宽表:", fn_pass)  # 打印真实测试池通过率宽表路径
    print("纯 posterior validation 表:", fn_posterior_only)  # 打印后验生成测试池通过率宽表路径
    print("好批次摘要表:", fn_good)  # 打印提示信息
    print("坏批次摘要表:", fn_bad)  # 打印提示信息

    print("\n===== 示例预览：强波动-好批次摘要表 =====")  # 打印提示信息
    print(good_out.head(20))  # 打印提示信息

    print("\n===== 示例预览：强波动-坏批次摘要表 =====")  # 打印提示信息
    print(bad_out.head(20))  # 打印提示信息


if __name__ == "__main__":  # 条件判断
    main()  # 执行本行逻辑
