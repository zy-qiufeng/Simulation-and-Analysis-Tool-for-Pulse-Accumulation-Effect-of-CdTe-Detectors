#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CdTe探测器脉冲堆积效应仿真与分析工具
=====================================
项目目标：
    模拟CdTe探测器在X射线照射下的脉冲堆积现象，实现脉冲信号去堆积与能谱重建，
    直观展示探测器性能对检测结果的影响。

主要功能模块：
    1. X射线源能谱模型
       - 连续轫致辐射谱（Bremsstrahlung），基于Kramers定律生成
       - 特征谱线（以钨靶W为例），叠加在连续谱上
    2. CdTe探测器响应模型
       - 能量分辨率（Fano因子 + 电子噪声展宽）
       - 能量依赖的探测效率（低能窗口吸收 + 高能穿透损失）
    3. 脉冲波形模型
       - 双指数脉冲（快速上升对应电子收集，慢速下降对应空穴收集）
    4. 脉冲堆积效应仿真
       - 蒙特卡洛方法模拟光子泊松到达过程
       - 检测时间间隔小于成形时间的堆积事件
    5. 去堆积校正算法
       - 迭代去卷积方法，逐步去除堆积贡献
    6. 可视化
       - 综合分析图（7个子图）
       - 能谱对比图（4个子图，突出连续谱成分）

依赖库：
    numpy, matplotlib, scipy

使用方法：
    python CdTe_detector_simulation.py

输出文件：
    CdTe_pileup_analysis.png      综合分析图
    CdTe_spectrum_comparison.png  能谱对比图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
import warnings

warnings.filterwarnings('ignore')

# 尝试设置中文字体（在有中文字体的系统上生效，否则使用默认字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei',
                                   'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 第一部分：X射线源能谱模型
# ============================================================

def generate_bremsstrahlung_spectrum(E_max, E_min=1.0, n_points=500):
    """
    生成连续轫致辐射（Bremsstrahlung）X射线能谱。

    物理基础：
        Kramers定律描述了管电压为 U（kV）、靶材原子序数为 Z 时，
        单位能量间隔内的 X 射线强度：
            I(E) = K * Z * (E_max / E - 1)
        其中 E_max = e*U 为最大光子能量，K 为比例常数。

        低能修正：低能光子被探测器窗口（铍窗）和空气强烈吸收，
        使用指数修正因子近似该效应。

    参数
    ----------
    E_max : float
        最大光子能量 (keV)，等于管电压数值（单位kV）
    E_min : float
        最低能量截断 (keV)，默认 1.0 keV
    n_points : int
        能量轴采样点数，默认 500

    返回
    -------
    energies : ndarray, shape (n_points,)
        能量轴 (keV)
    intensities : ndarray, shape (n_points,)
        归一化辐射强度（最大值为 1）
    """
    # 在 [E_min, E_max] 区间线性采样能量轴
    energies = np.linspace(E_min, E_max * 0.999, n_points)

    # 钨靶 (Z=74) 的 Kramers 公式
    Z = 74          # 钨靶原子序数
    K = 1.0         # 比例常数（用于归一化，数值不影响形状）
    intensities = K * Z * (E_max / energies - 1)

    # 低能端修正：铍窗和空气吸收导致低能光子强度显著衰减
    # 特征衰减能量取为 E_max 的 8%（经验参数，典型值约 5-10%）
    E_cutoff = E_max * 0.08
    absorption_correction = 1.0 - np.exp(-(energies - E_min) / E_cutoff)
    intensities *= absorption_correction

    # 防止数值负值，并归一化到 [0, 1]
    intensities = np.maximum(intensities, 0)
    if intensities.max() > 0:
        intensities /= intensities.max()

    return energies, intensities


def generate_characteristic_lines(tube_voltage=120):
    """
    生成钨靶 X 射线管的特征谱线列表。

    物理背景：
        当入射电子能量超过靶原子某壳层的结合能时，
        会将该壳层电子击出，外层电子向内跃迁并发射特征 X 射线。
        钨 (W, Z=74) 的主要特征谱系：
            - L 系：8~12 keV
            - K 系：57~70 keV（需要管电压 > 69.5 kV 才能激发）

    参数
    ----------
    tube_voltage : float
        管电压 (kV)，决定哪些特征谱线被激发（只激发
        能量小于管电压的谱线）

    返回
    -------
    lines : list of dict
        被激发的特征谱线列表，每个字典包含：
            'energy'    (float) : 谱线能量 (keV)
            'intensity' (float) : 相对强度（相对于最强线归一化）
            'name'      (str)   : 谱线标识（如 'W Kα1'）
    """
    # 钨靶完整特征谱线数据（能量 keV，相对强度，名称）
    # 数据来源：NIST X-ray transition energies database
    all_lines = [
        {'energy':  8.40, 'intensity': 0.15, 'name': 'W L\u03b1'},   # W Lα (L系)
        {'energy':  9.67, 'intensity': 0.08, 'name': 'W L\u03b2'},   # W Lβ (L系)
        {'energy': 11.29, 'intensity': 0.04, 'name': 'W L\u03b3'},   # W Lγ (L系)
        {'energy': 57.98, 'intensity': 0.35, 'name': 'W K\u03b12'},  # W Kα2 (K系)
        {'energy': 59.32, 'intensity': 0.60, 'name': 'W K\u03b11'},  # W Kα1 (K系，最强)
        {'energy': 67.24, 'intensity': 0.20, 'name': 'W K\u03b21'},  # W Kβ1 (K系)
        {'energy': 69.10, 'intensity': 0.06, 'name': 'W K\u03b22'},  # W Kβ2 (K系)
    ]

    # 只保留能量低于管电压的谱线（激发条件）
    active_lines = [line for line in all_lines if line['energy'] < tube_voltage]
    return active_lines


def generate_source_spectrum(E_max=120.0, n_channels=500, count_rate=1e5):
    """
    生成完整 X 射线源能谱（连续谱 + 特征谱线）。

    方法：
        1. 用 Kramers 公式生成连续轫致辐射谱（占总计数的 70%）
        2. 将特征谱线以极窄高斯 (σ=0.15 keV) 叠加到连续谱上（占 30%）
        3. 对结果施加泊松统计涨落噪声（模拟实际测量统计性）

    参数
    ----------
    E_max : float
        最大光子能量 (keV)，默认 120 keV（对应 120 kV 管电压）
    n_channels : int
        能谱道数，默认 500
    count_rate : float
        目标总计数率 (counts/s)，默认 1e5 cps

    返回
    -------
    energies : ndarray, shape (n_channels,)
        能量轴 (keV)
    spectrum : ndarray, shape (n_channels,)
        源能谱计数值（含泊松噪声）
    """
    # 建立能量轴（1 keV 起始，防止零点奇异）
    energies = np.linspace(1.0, E_max, n_channels)

    # ------ 连续轫致辐射谱 ------
    _, brem = generate_bremsstrahlung_spectrum(E_max, E_min=1.0, n_points=n_channels)

    # 连续谱占总计数的 70%
    continuous_fraction = 0.70
    spectrum = brem * (continuous_fraction * count_rate)

    # ------ 特征谱线叠加 ------
    char_lines = generate_characteristic_lines(E_max)
    if char_lines:
        # 特征谱线占总计数的 30%，按各线相对强度分配
        line_fraction = 1.0 - continuous_fraction
        total_rel_intensity = sum(ln['intensity'] for ln in char_lines)

        for line in char_lines:
            # 特征线固有宽度极窄，用 σ=0.15 keV 的高斯模拟
            sigma_line = 0.15   # keV，X 射线管特征线的固有宽度

            # 高斯归一化：在离散能量轴上计算高斯
            gauss = np.exp(-0.5 * ((energies - line['energy']) / sigma_line) ** 2)
            de = energies[1] - energies[0]                  # 道宽 (keV/道)
            gauss_area = sigma_line * np.sqrt(2 * np.pi)    # 连续高斯面积
            gauss /= (gauss_area / de)                      # 归一化为每道计数

            # 按相对强度分配计数
            rel_weight = line['intensity'] / total_rel_intensity
            spectrum += gauss * rel_weight * (line_fraction * count_rate)

    # 施加泊松噪声（真实探测器统计涨落）
    spectrum = np.random.poisson(np.maximum(spectrum, 0).astype(float)).astype(float)

    return energies, spectrum


# ============================================================
# 第二部分：CdTe 探测器模型
# ============================================================

class CdTeDetector:
    """
    CdTe（碲化镉）X 射线探测器物理模型。

    模拟特性：
        1. 能量分辨率
           - 统计涨落项：FWHM_stat = 2.355 * sqrt(F * w * E)
             F=Fano因子(≈0.1)，w=电子空穴对产生能量(4.43 eV)
           - 电子噪声项：FWHM_noise（固定值，约 0.5 keV）
           - 总 FWHM = sqrt(FWHM_stat² + FWHM_noise²)
        2. 探测效率
           - 低能端：铍窗吸收导致效率随能量降低而下降
           - 高能端：光子穿透 CdTe 晶体导致效率下降
             （用 Beer-Lambert 定律和简化的质量衰减系数描述）

    属性
    ----------
    thickness : float
        探测器厚度 (mm)
    bias_voltage : float
        偏置电压 (V)
    fano_factor : float
        Fano 因子，CdTe 约 0.1（描述电荷产生的统计涨落）
    pair_creation_energy : float
        每对电子-空穴对所需能量 (eV)，CdTe 约 4.43 eV
    electronic_noise : float
        电子学噪声折合为 FWHM (keV)，约 0.5 keV
    """

    def __init__(self, thickness=2.0, bias_voltage=500):
        """
        初始化 CdTe 探测器参数。

        参数
        ----------
        thickness : float
            探测器晶体厚度 (mm)，默认 2.0 mm
        bias_voltage : float
            偏置电压 (V)，默认 500 V
        """
        self.thickness = thickness
        self.bias_voltage = bias_voltage

        # CdTe 材料物理参数（文献值）
        self.fano_factor = 0.10          # Fano 因子（CdTe：~0.1，Si：~0.12）
        self.pair_creation_energy = 4.43  # 电子空穴对产生能量 (eV)，CdTe 约 4.43 eV
        self.electronic_noise = 0.50      # 电子学噪声 FWHM (keV)，约 0.5 keV

        # CdTe 密度和探测效率系数
        self.density = 5.85               # CdTe 密度 (g/cm³)
        self.base_efficiency = 0.95       # 基础探测效率（量子产率损失）

        print(f"CdTe 探测器初始化：厚度={thickness} mm，偏置电压={bias_voltage} V")

    def energy_resolution(self, energy_keV):
        """
        计算给定能量处的能量分辨率（FWHM）。

        公式：
            FWHM_stat  = 2.355 * sqrt(F * w * E)   [统计涨落项]
            FWHM_noise = 电子学噪声（常数）
            FWHM_total = sqrt(FWHM_stat² + FWHM_noise²)

        参数
        ----------
        energy_keV : float or ndarray
            入射光子能量 (keV)

        返回
        -------
        fwhm : float or ndarray
            总能量分辨率 FWHM (keV)
        """
        energy_keV = np.asarray(energy_keV, dtype=float)

        # 统计涨落引起的 FWHM（转换为 keV）
        energy_eV = energy_keV * 1000.0                                 # keV → eV
        n_pairs = energy_eV / self.pair_creation_energy                 # 产生的电子空穴对数
        fwhm_stat = 2.355 * np.sqrt(self.fano_factor * n_pairs) * \
                    (self.pair_creation_energy / 1000.0)                # eV → keV

        # 总 FWHM（正交叠加统计项和噪声项）
        fwhm_total = np.sqrt(fwhm_stat ** 2 + self.electronic_noise ** 2)
        return fwhm_total

    def detector_efficiency(self, energy_keV):
        """
        计算探测器在给定能量处的探测效率。

        模型：
            - 低能端：exp(-E_cutoff/E) 修正（铍窗 + 死层吸收）
            - 高能端：1 - exp(-μ·ρ·d)（Beer-Lambert 定律，光电效应为主）
              μ(E) ≈ μ₀ * (E/E₀)^(-2.7)   （简化光电截面能量依赖）

        参数
        ----------
        energy_keV : float or ndarray
            光子能量 (keV)

        返回
        -------
        efficiency : float or ndarray
            探测效率，取值范围 [0, 1]
        """
        energy_keV = np.asarray(energy_keV, dtype=float)

        # 低能端修正（铍窗和 CdTe 死层对低能光子的吸收）
        E_window_cutoff = 4.0   # 特征衰减能量 (keV)
        low_energy_factor = 1.0 - np.exp(-energy_keV / E_window_cutoff)

        # 高能端修正（光电效应截面随能量增大而减小）
        # CdTe 在 10 keV 处的质量衰减系数约 70 cm²/g
        mu0 = 70.0              # 参考质量衰减系数 (cm²/g) @10 keV
        E0 = 10.0               # 参考能量 (keV)
        alpha = 2.7             # 光电截面能量指数（理论值约 3，这里用 2.7 近似）
        mu = mu0 * (E0 / np.maximum(energy_keV, 0.1)) ** alpha  # 质量衰减系数

        thickness_cm = self.thickness * 0.1     # mm → cm
        high_energy_factor = 1.0 - np.exp(-mu * self.density * thickness_cm)

        # 总效率（低能 × 高能 × 基础效率）
        efficiency = self.base_efficiency * low_energy_factor * high_energy_factor
        return np.clip(efficiency, 0.0, 1.0)

    def apply_detector_response(self, energies, spectrum):
        """
        对理想入射能谱施加探测器响应（能量展宽 + 效率修正）。

        方法：
            对能谱中每个非零道，用该能量对应的高斯函数（宽度由
            energy_resolution 决定）对计数进行展宽，同时乘以该
            能量下的探测效率。

        参数
        ----------
        energies : ndarray, shape (n,)
            能量轴 (keV)
        spectrum : ndarray, shape (n,)
            输入理想能谱（道计数）

        返回
        -------
        detected_spectrum : ndarray, shape (n,)
            施加探测器响应后的能谱（道计数）
        """
        n = len(energies)
        de = energies[1] - energies[0]       # 道宽 (keV/道)
        detected_spectrum = np.zeros(n)
        channel_idx = np.arange(n)

        for i in range(n):
            if spectrum[i] <= 0:
                continue

            # 当前能量道对应的探测器参数
            fwhm = self.energy_resolution(energies[i])  # FWHM (keV)
            sigma_ch = (fwhm / 2.355) / de              # σ (道)
            eff = self.detector_efficiency(energies[i]) # 探测效率

            # 高斯响应函数（归一化，保证计数守恒）
            gauss = np.exp(-0.5 * ((channel_idx - i) / sigma_ch) ** 2)
            gauss_sum = gauss.sum()
            if gauss_sum > 0:
                gauss /= gauss_sum

            # 叠加到输出谱（计数 × 效率 × 归一化高斯）
            detected_spectrum += spectrum[i] * eff * gauss

        return detected_spectrum


# ============================================================
# 第三部分：脉冲波形模型
# ============================================================

def generate_pulse_shape(t, t0, amplitude, tau_rise=0.10, tau_fall=2.0):
    """
    生成 CdTe 探测器的单个脉冲波形（双指数模型）。

    物理背景：
        CdTe 探测器中，X 射线光子在晶体内产生电子-空穴对。
        由于 CdTe 中空穴迁移率远低于电子迁移率，脉冲呈现：
            - 快速上升前沿：由电子快速漂移到正极决定（τ_rise ~0.1 μs）
            - 缓慢下降后沿：由空穴缓慢漂移到负极决定（τ_fall ~2 μs）

        双指数脉冲模型（t ≥ t₀）：
            V(t) = A * [1 - exp(-(t-t₀)/τ_rise)] * exp(-(t-t₀)/τ_fall)

    参数
    ----------
    t : ndarray
        时间轴 (μs)
    t0 : float
        脉冲触发时刻 (μs)
    amplitude : float
        脉冲幅度（与入射光子能量成正比，单位 keV）
    tau_rise : float
        上升时间常数 (μs)，默认 0.10 μs
    tau_fall : float
        下降时间常数 (μs)，默认 2.0 μs

    返回
    -------
    pulse : ndarray, shape同 t
        脉冲电压波形（单位与 amplitude 相同）
    """
    pulse = np.zeros_like(t, dtype=float)
    mask = t >= t0
    dt = t[mask] - t0

    # 双指数波形：上升沿 × 下降沿
    pulse[mask] = amplitude * (1.0 - np.exp(-dt / tau_rise)) * np.exp(-dt / tau_fall)
    return pulse


# ============================================================
# 第四部分：脉冲堆积效应仿真
# ============================================================

class PileupSimulator:
    """
    脉冲堆积效应仿真器（蒙特卡洛方法）。

    物理背景：
        当 X 射线计数率较高时，相邻两个光子的到达时间间隔
        可能小于探测器的成形时间（脉冲持续时间）。此时两个
        脉冲在时域上发生重叠，幅度相加，系统将两个光子误判
        为一个能量为二者之和的"假光子"——这就是脉冲堆积效应。

        堆积概率（泊松过程近似）：
            P_pileup ≈ 1 - exp(-n * τ_s)
        其中 n 为计数率 (cps)，τ_s 为成形时间 (s)。

    属性
    ----------
    count_rate : float
        X 射线计数率 (cps)
    shaping_time : float
        探测器成形时间 (μs)
    pileup_probability : float
        理论堆积概率
    """

    def __init__(self, count_rate=1e5, shaping_time=2.0):
        """
        初始化堆积仿真器。

        参数
        ----------
        count_rate : float
            计数率 (cps)，默认 1e5 cps（高计数率场景）
        shaping_time : float
            成形时间 (μs)，默认 2.0 μs
        """
        self.count_rate = count_rate
        self.shaping_time = shaping_time    # μs

        # 计算理论堆积概率（任意两个相邻脉冲发生堆积的概率）
        tau_s = shaping_time * 1e-6         # μs → s
        self.pileup_probability = 1.0 - np.exp(-count_rate * tau_s)

        print(f"堆积仿真器：计数率={count_rate:.1e} cps，"
              f"成形时间={shaping_time} μs，"
              f"理论堆积概率={self.pileup_probability:.2%}")

    def simulate_pileup(self, energies, spectrum, n_photons=20000):
        """
        蒙特卡洛模拟脉冲堆积过程。

        算法：
            1. 按输入能谱的概率分布随机抽取 n_photons 个光子能量
            2. 用指数分布随机生成相邻光子的到达时间间隔
               (平均间隔 = 1/count_rate)
            3. 顺序检查每对相邻脉冲：
               若时间间隔 < shaping_time，则判定为堆积事件：
                   - 记录能量 = E_i + E_{i+1}（幅度叠加）
                   - 跳过下一个脉冲（已被"吸收"进堆积事件）
               否则，记录正常测量能量 = E_i

        参数
        ----------
        energies : ndarray
            能量轴 (keV)
        spectrum : ndarray
            输入能谱（道计数，用作光子能量的概率权重）
        n_photons : int
            模拟总光子数，默认 20000

        返回
        -------
        measured_energies : ndarray
            测量到的光子能量列表（含堆积误判后的能量）
        true_energies : ndarray
            真实光子能量列表（理想无堆积情况）
        pileup_info : dict
            堆积统计信息，包含：
                'total_photons'   : 模拟光子总数
                'pileup_events'   : 堆积事件数
                'pileup_fraction' : 堆积比例
                'measured_photons': 最终记录的事件总数
        """
        # 按能谱计数归一化为概率分布
        weights = np.maximum(spectrum, 0).astype(float)
        weights_sum = weights.sum()
        if weights_sum == 0:
            raise ValueError("输入能谱全为零，无法生成光子能量样本。")
        weights /= weights_sum

        # 按概率抽取真实光子能量
        true_energies = np.random.choice(energies, size=n_photons, p=weights)

        # 生成泊松到达过程（指数分布的时间间隔）
        mean_interval_us = 1.0 / self.count_rate * 1e6   # 平均间隔 (μs)
        intervals = np.random.exponential(mean_interval_us, size=n_photons)
        arrival_times = np.cumsum(intervals)              # 累积到达时刻 (μs)

        # 逐事件检测堆积
        measured_energies = []
        pileup_count = 0
        i = 0
        skip_next = False

        while i < n_photons:
            if skip_next:
                # 该光子已被前一个堆积事件"吸收"，跳过
                skip_next = False
                i += 1
                continue

            if i < n_photons - 1:
                dt = arrival_times[i + 1] - arrival_times[i]   # 相邻脉冲时间间隔 (μs)
                if dt < self.shaping_time:
                    # 堆积：两脉冲幅度相加，系统无法区分
                    measured_energies.append(true_energies[i] + true_energies[i + 1])
                    pileup_count += 1
                    skip_next = True    # 下一个光子已参与堆积，跳过
                else:
                    measured_energies.append(true_energies[i])
            else:
                # 最后一个光子，不存在后续脉冲，直接记录
                measured_energies.append(true_energies[i])

            i += 1

        pileup_info = {
            'total_photons':    n_photons,
            'pileup_events':    pileup_count,
            'pileup_fraction':  pileup_count / n_photons,
            'measured_photons': len(measured_energies),
        }

        return np.array(measured_energies), true_energies, pileup_info


# ============================================================
# 第五部分：去堆积校正算法
# ============================================================

def depileup_spectrum(pileup_spectrum, energies, iterations=12):
    """
    迭代去堆积（de-pileup）校正算法。

    原理：
        堆积测量谱 M(E) 可分解为：
            M(E) = S(E) + p * [S ⊗ S](E) + O(p²)
        其中 S(E) 为真实谱，p 为堆积概率，⊗ 表示卷积。

        迭代策略（每轮）：
            1. 用当前估计谱 S_k 计算堆积贡献：
               C_k = p * [S_k_norm ⊗ S_k_norm] * ||M||
            2. 更新估计：S_{k+1} = max(M - C_k, 0)
            3. 可选轻度高斯平滑，防止数值振荡

        同时在每轮动态更新堆积概率估计 p，
        基于测量谱与校正谱总计数之差来估算。

    参数
    ----------
    pileup_spectrum : ndarray
        含堆积效应的测量能谱
    energies : ndarray
        能量轴 (keV)
    iterations : int
        迭代次数，默认 12

    返回
    -------
    corrected : ndarray
        去堆积校正后的能谱（非负）
    """
    n = len(pileup_spectrum)
    corrected = pileup_spectrum.copy().astype(float)

    # 初始堆积概率估计（保守值）
    pileup_prob = 0.08

    for k in range(iterations):
        total_counts = corrected.sum()
        if total_counts <= 0:
            break

        # 归一化当前估计谱为概率密度
        normalized = corrected / total_counts

        # 计算堆积贡献谱：真实谱的自卷积（能量加法对应离散卷积）
        conv = np.convolve(normalized, normalized)[:n]
        pileup_contribution = conv * total_counts * pileup_prob

        # 从测量谱中减去估计的堆积贡献
        new_corrected = pileup_spectrum - pileup_contribution
        new_corrected = np.maximum(new_corrected, 0.0)    # 保证非负

        # 前几轮施加轻度高斯平滑（防止数值噪声放大），最后几轮不平滑
        if k < iterations - 3:
            new_corrected = gaussian_filter1d(new_corrected, sigma=0.6)

        # 动态更新堆积概率估计
        corrected_total = new_corrected.sum()
        measured_total = pileup_spectrum.sum()
        if measured_total > 0 and corrected_total > 0:
            # 堆积事件减少了总事件数，用计数差估计堆积概率
            pileup_prob = min(0.35, max(0.01,
                (measured_total - corrected_total) / measured_total * 1.5))

        corrected = new_corrected

    return corrected


# ============================================================
# 第六部分：可视化函数
# ============================================================

def plot_complete_analysis(energies, source_spectrum, detected_spectrum,
                           pileup_spectrum, corrected_spectrum,
                           pileup_info, detector):
    """
    绘制综合分析图（7个子图）。

    子图布局（3行×3列网格）：
        [0,0:2] 行0 左两列：X 射线源能谱（连续谱 + 特征线，主图）
        [0,2]   行0 右一列：CdTe 探测器能量分辨率曲线
        [1,0]   行1 左一列：探测器响应后的理想能谱（无堆积）
        [1,1]   行1 中一列：含堆积效应的测量能谱
        [1,2]   行1 右一列：去堆积校正后能谱
        [2,0:2] 行2 左两列：脉冲波形与堆积示意图
        [2,2]   行2 右一列：探测器探测效率曲线

    参数
    ----------
    energies : ndarray
        能量轴 (keV)
    source_spectrum : ndarray
        X 射线源能谱
    detected_spectrum : ndarray
        探测器响应后能谱（无堆积）
    pileup_spectrum : ndarray
        含堆积效应的测量能谱
    corrected_spectrum : ndarray
        去堆积校正后能谱
    pileup_info : dict
        堆积统计信息
    detector : CdTeDetector
        探测器对象
    """
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    # 颜色方案
    C = {
        'continuous': '#5B9BD5',    # 蓝色：连续谱
        'source':     '#1F4E79',    # 深蓝：完整源谱
        'char_annot': '#C00000',    # 深红：特征线标注
        'detected':   '#70AD47',    # 绿色：探测器响应谱
        'pileup':     '#FF4444',    # 红色：堆积谱
        'corrected':  '#8B008B',    # 紫色：校正谱
        'resolution': '#ED7D31',    # 橙色：分辨率
        'efficiency': '#00B0A0',    # 青色：效率
    }

    E_max = energies[-1]
    # 获取纯连续谱成分（用于图中叠加显示）
    _, brem = generate_bremsstrahlung_spectrum(E_max, E_min=energies[0],
                                               n_points=len(energies))
    # 将连续谱缩放到源谱连续成分的量级（连续谱占 70%）
    continuous_component = brem / brem.max() * source_spectrum.max() * 0.70

    # ────────────────────────────────────────────────────────
    # 子图 1：X 射线源能谱（连续谱 + 特征谱线，主展示图）
    # ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])

    # 1a. 填充连续轫致辐射谱（面积填充，突出连续谱形态）
    ax1.fill_between(energies, 0, continuous_component,
                     alpha=0.45, color=C['continuous'],
                     label='连续轫致辐射谱 (Bremsstrahlung)')
    ax1.plot(energies, continuous_component,
             color=C['continuous'], linewidth=1.8, alpha=0.9)

    # 1b. 绘制完整源谱（连续 + 特征线叠加）
    ax1.plot(energies, source_spectrum,
             color=C['source'], linewidth=1.5, alpha=0.85,
             label='完整 X 射线谱（连续谱 + 特征谱线）')

    # 1c. 标注特征谱线
    char_lines = generate_characteristic_lines(E_max)
    for line in char_lines:
        idx = np.argmin(np.abs(energies - line['energy']))
        peak_val = source_spectrum[idx]
        if peak_val > source_spectrum.max() * 0.03:   # 只标注较强的线
            ax1.annotate(
                line['name'],
                xy=(energies[idx], peak_val),
                xytext=(energies[idx] + 3, peak_val * 1.12),
                fontsize=7.5, color=C['char_annot'],
                arrowprops=dict(arrowstyle='->', color=C['char_annot'], lw=0.9))

    ax1.set_xlabel('能量 (keV)', fontsize=11)
    ax1.set_ylabel('计数', fontsize=11)
    ax1.set_title('X 射线源能谱\n（连续轫致辐射谱 + 钨靶特征谱线）',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_xlim(energies[0], E_max)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────────
    # 子图 2：CdTe 探测器能量分辨率随能量的变化
    # ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])

    test_E = np.linspace(5, E_max, 300)
    fwhm = detector.energy_resolution(test_E)
    rel_fwhm_pct = fwhm / test_E * 100     # 相对分辨率 (%)

    ax2.plot(test_E, rel_fwhm_pct,
             color=C['resolution'], linewidth=2)
    ax2.fill_between(test_E, 0, rel_fwhm_pct,
                     alpha=0.3, color=C['resolution'])

    # 标注几个典型能量点的分辨率数值
    for E_mark in [10, 30, 59, 100]:
        if E_mark < E_max:
            rv = detector.energy_resolution(E_mark) / E_mark * 100
            ax2.plot(E_mark, rv, 'o', color=C['resolution'], markersize=5)
            ax2.annotate(f'{rv:.1f}%',
                         xy=(E_mark, rv), xytext=(E_mark + 4, rv + 0.3),
                         fontsize=7.5, color='darkred')

    ax2.set_xlabel('能量 (keV)', fontsize=11)
    ax2.set_ylabel('FWHM / E  (%)', fontsize=11)
    ax2.set_title('CdTe 探测器能量分辨率\n（FWHM/E vs 能量）',
                  fontsize=11, fontweight='bold')
    ax2.set_xlim(0, E_max)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────────
    # 子图 3：探测器响应后的理想能谱（低计数率，无堆积）
    # ────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    # 连续谱成分（探测器响应后，用平滑近似）
    detected_continuous_approx = gaussian_filter1d(detected_spectrum, sigma=6) * 0.75

    ax3.fill_between(energies, 0, detected_continuous_approx,
                     alpha=0.30, color='#AAAAAA',
                     label='连续谱成分（估计）')
    ax3.fill_between(energies, detected_continuous_approx, detected_spectrum,
                     alpha=0.35, color=C['detected'],
                     label='特征峰成分')
    ax3.plot(energies, detected_spectrum,
             color=C['detected'], linewidth=1.8,
             label='探测器响应谱（无堆积）')

    ax3.set_xlabel('能量 (keV)', fontsize=10)
    ax3.set_ylabel('计数', fontsize=10)
    ax3.set_title('探测器响应能谱\n（低计数率，无堆积效应）',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=7.5)
    ax3.set_xlim(energies[0], E_max)
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────────
    # 子图 4：含堆积效应的测量能谱
    # ────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.fill_between(energies, 0, pileup_spectrum,
                     alpha=0.30, color=C['pileup'])
    ax4.plot(energies, pileup_spectrum,
             color=C['pileup'], linewidth=1.8,
             label='高计数率测量谱（含堆积）')

    # 叠加无堆积参考谱（按最大值缩放便于形状比较）
    if detected_spectrum.max() > 0 and pileup_spectrum.max() > 0:
        scale = pileup_spectrum.max() / detected_spectrum.max()
        ax4.plot(energies, detected_spectrum * scale,
                 color=C['detected'], linewidth=1.2, linestyle='--',
                 alpha=0.65, label='无堆积参考谱（等比缩放）')

    # 在图内标注堆积统计数据
    ax4.text(0.97, 0.95,
             f"堆积比例: {pileup_info['pileup_fraction']:.1%}\n"
             f"堆积事件: {pileup_info['pileup_events']}",
             transform=ax4.transAxes, fontsize=9,
             va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', alpha=0.9))

    ax4.set_xlabel('能量 (keV)', fontsize=10)
    ax4.set_ylabel('计数', fontsize=10)
    ax4.set_title('高计数率测量能谱\n（含脉冲堆积效应）',
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7.5)
    ax4.set_xlim(energies[0], E_max)
    ax4.set_ylim(bottom=0)
    ax4.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────────
    # 子图 5：去堆积校正后能谱与参考谱对比
    # ────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])

    ax5.fill_between(energies, 0, corrected_spectrum,
                     alpha=0.30, color=C['corrected'])
    ax5.plot(energies, corrected_spectrum,
             color=C['corrected'], linewidth=1.8,
             label='去堆积校正谱')

    # 参考谱（按相同总计数缩放）
    corrected_total = corrected_spectrum.sum()
    detected_total = detected_spectrum.sum()
    if detected_total > 0 and corrected_total > 0:
        ref_scale = corrected_total / detected_total
        ax5.plot(energies, detected_spectrum * ref_scale,
                 color=C['detected'], linewidth=1.2, linestyle='--',
                 alpha=0.7, label='无堆积参考谱（等计数缩放）')

    ax5.set_xlabel('能量 (keV)', fontsize=10)
    ax5.set_ylabel('计数', fontsize=10)
    ax5.set_title('去堆积校正能谱\n（与无堆积参考谱对比）',
                  fontsize=11, fontweight='bold')
    ax5.legend(fontsize=7.5)
    ax5.set_xlim(energies[0], E_max)
    ax5.set_ylim(bottom=0)
    ax5.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────────
    # 子图 6：CdTe 探测器脉冲波形与堆积示意图
    # ────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])

    t = np.linspace(0, 22, 2200)   # 时间轴 0~22 μs，分辨率 0.01 μs

    # 场景 A：孤立脉冲（能量 59.3 keV，对应 W Kα1 线）
    p_isolated_59 = generate_pulse_shape(t, t0=2.0, amplitude=59.3)
    # 场景 B：另一个孤立脉冲（能量 30 keV），作为参考
    p_isolated_30 = generate_pulse_shape(t, t0=14.0, amplitude=30.0)
    # 场景 C：发生堆积的两个脉冲（时间间隔 1.0 μs < 成形时间 2.0 μs）
    p_a = generate_pulse_shape(t, t0=6.0, amplitude=59.3)   # 脉冲 A（59 keV）
    p_b = generate_pulse_shape(t, t0=7.0, amplitude=30.0)   # 脉冲 B（30 keV）
    p_pileup = p_a + p_b                                     # 堆积叠加波形

    # 绘制各脉冲
    ax6.plot(t, p_isolated_59, color='royalblue', linewidth=2.0,
             label='孤立脉冲（59.3 keV，W Kα1）')
    ax6.plot(t, p_a, color='limegreen', linewidth=1.5, linestyle='--',
             alpha=0.75, label='脉冲 A - 真实（59 keV，堆积场景）')
    ax6.plot(t, p_b, color='darkorange', linewidth=1.5, linestyle='--',
             alpha=0.75, label='脉冲 B - 真实（30 keV，堆积场景）')
    ax6.plot(t, p_pileup, color='crimson', linewidth=2.5,
             label=f'堆积叠加波形（测量幅度≈{p_pileup.max():.0f} keV，误判！）')
    ax6.plot(t, p_isolated_30, color='mediumpurple', linewidth=1.5,
             alpha=0.85, label='孤立脉冲（30 keV，参考）')

    # 标注堆积区域（浅红背景）
    ax6.axvspan(5.5, 11.5, alpha=0.10, color='red')
    ax6.annotate('脉冲堆积区域\n间隔(1μs) < 成形时间(2μs)',
                 xy=(8.5, p_pileup.max() * 0.85),
                 xytext=(10.5, p_pileup.max() * 0.95),
                 fontsize=9, color='crimson',
                 arrowprops=dict(arrowstyle='->', color='crimson', lw=1.0))

    # 标注成形时间（箭头跨度 = 2 μs = 成形时间）
    y_ref = -4
    ax6.annotate('', xy=(8.0, y_ref), xytext=(6.0, y_ref),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax6.text(7.0, y_ref + 0.5, '成形时间\n2μs', ha='center', va='bottom',
             fontsize=8, color='gray')

    ax6.axhline(0, color='black', linewidth=0.5)
    ax6.set_xlabel('时间 (μs)', fontsize=11)
    ax6.set_ylabel('脉冲幅度 (keV)', fontsize=11)
    ax6.set_title('CdTe 探测器脉冲波形与堆积效应示意\n'
                  '（成形时间 = 2 μs；双指数波形：τ_rise=0.1μs，τ_fall=2.0μs）',
                  fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8, loc='upper right', ncol=2)
    ax6.set_xlim(0, 22)
    ax6.set_ylim(-6, p_pileup.max() * 1.35)
    ax6.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────────
    # 子图 7：探测器探测效率随能量的变化
    # ────────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])

    eff = detector.detector_efficiency(test_E) * 100    # 转换为百分比

    ax7.plot(test_E, eff, color=C['efficiency'], linewidth=2)
    ax7.fill_between(test_E, 0, eff, alpha=0.3, color=C['efficiency'])

    # 标注峰值效率
    peak_idx = np.argmax(eff)
    ax7.annotate(f'峰值: {eff[peak_idx]:.1f}%\n@{test_E[peak_idx]:.0f} keV',
                 xy=(test_E[peak_idx], eff[peak_idx]),
                 xytext=(test_E[peak_idx] + 8, eff[peak_idx] - 15),
                 fontsize=8, color='darkcyan',
                 arrowprops=dict(arrowstyle='->', color='darkcyan', lw=0.9))

    ax7.set_xlabel('能量 (keV)', fontsize=10)
    ax7.set_ylabel('探测效率 (%)', fontsize=10)
    ax7.set_title('CdTe 探测器探测效率\n（厚度 2 mm，偏压 500 V）',
                  fontsize=11, fontweight='bold')
    ax7.set_xlim(0, E_max)
    ax7.set_ylim(0, 105)
    ax7.grid(True, alpha=0.3)

    # 总标题
    fig.suptitle('CdTe 探测器脉冲堆积效应仿真与分析\n'
                 'Simulation and Analysis of Pulse Pile-up Effects in CdTe Detectors',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.savefig('CdTe_pileup_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.show()
    print("  综合分析图已保存：CdTe_pileup_analysis.png")
    return fig


def plot_spectrum_comparison(energies, source_spectrum, detected_spectrum,
                             pileup_spectrum, corrected_spectrum):
    """
    绘制能谱对比图（4个子图），重点展示连续 X 射线谱成分。

    子图布局（2行×2列）：
        [0,0] 源谱分解：连续谱（填充）+ 特征线叠加（颜色区分）
        [0,1] 探测器响应谱：连续谱成分与特征峰分别填充
        [1,0] 堆积效应影响：无堆积 vs 有堆积对比 + 堆积增量区域
        [1,1] 去堆积效果：堆积谱 → 校正谱 → 参考谱三者叠加对比

    参数
    ----------
    energies : ndarray
        能量轴 (keV)
    source_spectrum : ndarray
        X 射线源能谱
    detected_spectrum : ndarray
        探测器响应后能谱（无堆积参考）
    pileup_spectrum : ndarray
        含堆积效应的测量能谱
    corrected_spectrum : ndarray
        去堆积校正后能谱
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('能谱对比分析\n（连续 X 射线谱成分在各阶段的演变）',
                 fontsize=14, fontweight='bold')

    E_max = energies[-1]

    # 提取连续谱成分（从源谱生成）
    _, brem = generate_bremsstrahlung_spectrum(E_max, E_min=energies[0],
                                               n_points=len(energies))
    continuous_component = brem / brem.max() * source_spectrum.max() * 0.70

    # ────────────────────────────────────────────────────────
    # 子图 [0,0]：X 射线源能谱分解
    # ────────────────────────────────────────────────────────
    ax = axes[0, 0]

    # 连续谱（蓝色填充）
    ax.fill_between(energies, 0, continuous_component,
                    alpha=0.55, color='#5B9BD5',
                    label='连续轫致辐射谱')
    ax.plot(energies, continuous_component,
            color='steelblue', linewidth=2.0)

    # 特征谱线成分（在连续谱上方叠加，橙红色）
    char_above = np.maximum(source_spectrum - continuous_component, 0)
    ax.fill_between(energies, continuous_component,
                    continuous_component + char_above,
                    alpha=0.65, color='#FF7043',
                    label='特征谱线成分')

    # 完整源谱轮廓
    ax.plot(energies, source_spectrum,
            color='#1F4E79', linewidth=1.5, alpha=0.8,
            label='完整 X 射线谱')

    ax.set_xlabel('能量 (keV)', fontsize=11)
    ax.set_ylabel('计数', fontsize=11)
    ax.set_title('X 射线源能谱分解\n（蓝色=连续谱，橙色=特征谱线）', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(energies[0], E_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────────
    # 子图 [0,1]：探测器响应后能谱（连续谱 + 特征峰分层显示）
    # ────────────────────────────────────────────────────────
    ax = axes[0, 1]

    # 用高斯平滑近似探测谱中的连续谱底座
    detected_baseline = gaussian_filter1d(detected_spectrum, sigma=8) * 0.72

    ax.fill_between(energies, 0, detected_baseline,
                    alpha=0.40, color='#AAAAAA',
                    label='连续谱成分（平滑估计）')
    ax.fill_between(energies, detected_baseline, detected_spectrum,
                    alpha=0.45, color='#70AD47',
                    label='特征峰成分')
    ax.plot(energies, detected_spectrum,
            color='darkgreen', linewidth=2.0,
            label='探测器响应谱（无堆积）')

    ax.set_xlabel('能量 (keV)', fontsize=11)
    ax.set_ylabel('计数', fontsize=11)
    ax.set_title('探测器响应能谱\n（灰色=连续谱底座，绿色=特征峰）', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(energies[0], E_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────────
    # 子图 [1,0]：堆积效应的影响（归一化对比）
    # ────────────────────────────────────────────────────────
    ax = axes[1, 0]

    # 归一化（便于形状比较）
    norm_det = detected_spectrum / max(detected_spectrum.max(), 1e-9)
    norm_pile = pileup_spectrum / max(pileup_spectrum.max(), 1e-9)

    ax.fill_between(energies, 0, norm_det,
                    alpha=0.30, color='#70AD47',
                    label='无堆积（参考）')
    ax.fill_between(energies, 0, norm_pile,
                    alpha=0.25, color='#FF4444',
                    label='有堆积效应')
    ax.plot(energies, norm_det, color='darkgreen', linewidth=1.8)
    ax.plot(energies, norm_pile, color='crimson', linewidth=1.8)

    # 堆积新增计数（橙色高亮：堆积谱超出无堆积谱的部分）
    pile_excess = np.maximum(norm_pile - norm_det, 0)
    ax.fill_between(energies, norm_det, norm_det + pile_excess,
                    alpha=0.45, color='#FFB300',
                    label='堆积增加的计数（高能段假事件）')

    ax.set_xlabel('能量 (keV)', fontsize=11)
    ax.set_ylabel('归一化计数', fontsize=11)
    ax.set_title('堆积效应影响\n（黄色区域=堆积产生的虚假高能计数）', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(energies[0], E_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # ────────────────────────────────────────────────────────
    # 子图 [1,1]：去堆积校正效果三谱对比
    # ────────────────────────────────────────────────────────
    ax = axes[1, 1]

    # 三谱均归一化
    norm_pile2 = pileup_spectrum / max(pileup_spectrum.max(), 1e-9)
    norm_corr = corrected_spectrum / max(corrected_spectrum.max(), 1e-9)
    norm_det2 = detected_spectrum / max(detected_spectrum.max(), 1e-9)

    ax.fill_between(energies, 0, norm_corr,
                    alpha=0.30, color='#8B008B')
    ax.plot(energies, norm_pile2, color='crimson', linewidth=1.5,
            alpha=0.65, label='堆积测量谱')
    ax.plot(energies, norm_corr, color='#8B008B', linewidth=2.2,
            label='去堆积校正谱')
    ax.plot(energies, norm_det2, color='darkgreen', linewidth=1.8,
            linestyle='--', alpha=0.80,
            label='无堆积理想谱（参考）')

    # 用箭头示意校正方向
    ax.annotate('', xy=(E_max * 0.70, 0.35), xytext=(E_max * 0.70, 0.55),
                arrowprops=dict(arrowstyle='->', color='gray',
                                lw=1.5, connectionstyle='arc3'))
    ax.text(E_max * 0.72, 0.44, '堆积\n校正', fontsize=8,
            color='gray', ha='left', va='center')

    ax.set_xlabel('能量 (keV)', fontsize=11)
    ax.set_ylabel('归一化计数', fontsize=11)
    ax.set_title('去堆积校正效果\n（红→紫=校正谱，绿虚=理想参考谱）', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(energies[0], E_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('CdTe_spectrum_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.show()
    print("  能谱对比图已保存：CdTe_spectrum_comparison.png")
    return fig


# ============================================================
# 第七部分：主程序
# ============================================================

def main():
    """
    主函数：运行完整的 CdTe 探测器脉冲堆积效应仿真分析流程。

    执行步骤：
        1. 设置仿真参数（管电压、道数、计数率、成形时间、光子数）
        2. 生成 X 射线源能谱（连续谱 + 特征谱线）
        3. 初始化 CdTe 探测器模型
        4. 对理想能谱施加探测器响应（能量展宽 + 效率修正）
        5. 蒙特卡洛模拟脉冲堆积效应
        6. 运行迭代去堆积校正算法
        7. 绘制综合分析图和能谱对比图

    返回
    -------
    results : dict
        包含各阶段能谱数组和堆积统计信息
    """
    print("=" * 62)
    print("  CdTe 探测器脉冲堆积效应仿真与分析工具")
    print("  Simulation & Analysis Tool for CdTe Detector Pile-up")
    print("=" * 62)

    # ---- 设置随机种子（保证结果可重复）----
    np.random.seed(42)

    # ---- 仿真参数 ----
    E_MAX = 120.0       # 最大光子能量 (keV)，对应 120 kV 管电压
    N_CHANNELS = 500    # 能谱道数
    COUNT_RATE = 1e5    # 计数率 (cps)，高计数率以显示堆积效应
    SHAPING_TIME = 2.0  # 成形时间 (μs)
    N_PHOTONS = 25000   # 蒙特卡洛模拟光子数

    print(f"\n  仿真参数:")
    print(f"    管电压（最大光子能量）: {E_MAX:.0f} keV")
    print(f"    能谱道数            : {N_CHANNELS}")
    print(f"    计数率              : {COUNT_RATE:.1e} cps")
    print(f"    成形时间            : {SHAPING_TIME} μs")
    print(f"    模拟光子数          : {N_PHOTONS}")

    # ── 步骤 1：生成 X 射线源能谱 ──────────────────────────────
    print("\n[步骤 1] 生成 X 射线源能谱（连续谱 + 特征谱线）...")
    energies, source_spectrum = generate_source_spectrum(
        E_max=E_MAX, n_channels=N_CHANNELS, count_rate=COUNT_RATE)
    print(f"    能量范围: {energies[0]:.1f} ~ {energies[-1]:.1f} keV")
    print(f"    总计数  : {source_spectrum.sum():.0f}")

    # ── 步骤 2：初始化 CdTe 探测器 ─────────────────────────────
    print("\n[步骤 2] 初始化 CdTe 探测器模型...")
    detector = CdTeDetector(thickness=2.0, bias_voltage=500)
    print(f"    Fano 因子       : {detector.fano_factor}")
    print(f"    电子噪声 (FWHM) : {detector.electronic_noise} keV")

    # ── 步骤 3：应用探测器响应 ──────────────────────────────────
    print("\n[步骤 3] 应用探测器响应（能量分辨率展宽 + 探测效率）...")
    print("    （计算中，请稍候...）")
    detected_spectrum = detector.apply_detector_response(energies, source_spectrum)
    print(f"    响应后总计数: {detected_spectrum.sum():.0f}")
    for E_test in [10, 30, 59, 100]:
        if E_test < E_MAX:
            fwhm = detector.energy_resolution(E_test)
            print(f"    分辨率 @{E_test:3.0f} keV: FWHM = {fwhm:.2f} keV "
                  f"({fwhm / E_test * 100:.1f}%)")

    # ── 步骤 4：模拟脉冲堆积效应 ──────────────────────────────
    print(f"\n[步骤 4] 蒙特卡洛模拟脉冲堆积（{COUNT_RATE:.0e} cps）...")
    pileup_sim = PileupSimulator(count_rate=COUNT_RATE,
                                  shaping_time=SHAPING_TIME)
    measured_energies, true_energies, pileup_info = pileup_sim.simulate_pileup(
        energies, detected_spectrum, n_photons=N_PHOTONS)

    print(f"    模拟光子数: {pileup_info['total_photons']}")
    print(f"    堆积事件数: {pileup_info['pileup_events']}")
    print(f"    堆积比例  : {pileup_info['pileup_fraction']:.2%}")

    # 将测量能量列表转换为道计数直方图
    pileup_spectrum, _ = np.histogram(
        measured_energies, bins=N_CHANNELS,
        range=(energies[0], energies[-1]))
    pileup_spectrum = pileup_spectrum.astype(float)

    # ── 步骤 5：去堆积校正 ────────────────────────────────────
    print("\n[步骤 5] 运行迭代去堆积校正算法...")
    corrected_spectrum = depileup_spectrum(pileup_spectrum, energies, iterations=12)
    print(f"    校正前总计数: {pileup_spectrum.sum():.0f}")
    print(f"    校正后总计数: {corrected_spectrum.sum():.0f}")

    # ── 步骤 6：绘制结果图 ────────────────────────────────────
    print("\n[步骤 6] 绘制结果图...")
    print("  绘制综合分析图...")
    plot_complete_analysis(energies, source_spectrum, detected_spectrum,
                           pileup_spectrum, corrected_spectrum,
                           pileup_info, detector)

    print("  绘制能谱对比图...")
    plot_spectrum_comparison(energies, source_spectrum, detected_spectrum,
                             pileup_spectrum, corrected_spectrum)

    print("\n" + "=" * 62)
    print("  仿真完成！输出文件：")
    print("    CdTe_pileup_analysis.png      综合分析图（7子图）")
    print("    CdTe_spectrum_comparison.png  能谱对比图（4子图）")
    print("=" * 62)

    return {
        'energies':           energies,
        'source_spectrum':    source_spectrum,
        'detected_spectrum':  detected_spectrum,
        'pileup_spectrum':    pileup_spectrum,
        'corrected_spectrum': corrected_spectrum,
        'pileup_info':        pileup_info,
    }


if __name__ == '__main__':
    results = main()
