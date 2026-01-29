1#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import shutil
import imageio.v2 as imageio
import re
import math
import sys
import time
import gc
from tqdm import tqdm

# matplotlibのフォントをArialに設定
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
# LaTeXモードを無効化してフォントを確実にArialにする
plt.rcParams['text.usetex'] = False
# 数学テキストフォントもArialに設定（boldもArial）
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['mathtext.default'] = 'regular'

# ==================================================================
# 【変更可能な設定値セクション】
# ==================================================================

# ------------------------------------------------------------------
# 1. 実行モード設定
# ------------------------------------------------------------------
# True: 速度優先モード（高速化設定）
# False: 品質優先モード（高精度設定）
SPEED_MODE = True

# True: シンプル描画モード（タイトル、ボックス、凡例を非表示、プロットと軸目盛り・軸タイトルのみ表示）
SIMPLE_PLOT_MODE = False

# True: 単位変換を適用する（P: [uC/cm^2] → [C/m^2], R: 角度定義 → 変位定義）
# False: 単位変換を適用しない（元の単位のまま）
APPLY_UNIT_CONVERSION = True

# ----------- -------------------------------------------------------
# 2. 物理パラメータ（自由エネルギー計算の係数）
# ------------------------------------------------------------------
# Rの単位変換用の格子定数
LATTICE_CONSTANT_L = 0.4  # 格子定数l
# Rの単位変換: 角度定義R（φ）→ 変位定義r = l*φ/2 = 0.4*R/2 = 0.2*R
# つまり、R = r/0.2 = 5*r
# 変換係数: R^2 → 25*r^2, R^4 → 625*r^4, R^6 → 15625*r^6

fixed_params = { 
    'T_FIXED': -0,      # 固定する温度
    't_{cP}': 5,         # P臨界温度
    't_{cR}': 10,        # R臨界温度
    'alpha_1': 0.5,      # P^2項の係数  
    'alpha_2': 0.02,    # P^4項の係数
    'alpha_3': 0.0001, # P^6項の係数 
    'beta_1': 8,         # R^2項の係数  
    'beta_2': -60,      # R^4項の係数
    'beta_3': 12,       # R^6項の係数
    'gamma_22': -0.01,        # γ_22*P^2*R^2項の係数
    'gamma_42': 0.001,          # γ_42*P^4*R^2項の係数　
    'gamma_82': 0.0000,            # γ_82*P^8*R^2項の係数（新規追加）
    'gamma_64': -0.000000,      # γ_64*P^6*R^4項の係数
    'gamma_84': 0.0000000005,   # γ_84*P^8*R^4項の係数
    'gamma_86': 0.00000000000,  # γ_86*P^8*R^6項の係数
}

# ------------------------------------------------------------------
# 3. プロット生成制御フラグ
# ------------------------------------------------------------------
GENERATE_2D_CONTOUR = True      # 2Dコンター図を生成するか
GENERATE_3D_SURFACE = True      # 3Dサーフェス図を生成するか
GENERATE_CROSS_SECTION_P = False # P軸に対する垂直な断面プロットを生成するか
GENERATE_CROSS_SECTION_R = False  # R軸に対する垂直な断面プロットを生成するか
GENERATE_HYSTERESIS_ANIMATION =  False #not SPEED_MODE  # ヒステリシスループのGIFアニメーションを生成する（描画モード: True, スピードモード: False）
GENERATE_SUMMARY_GIF = False     # 全プロットを一覧表示するGIFを生成するか
GENERATE_3D_UNITCELL = False      # 3DユニットセルモデルのGIFアニメーションを生成するか
SHOW_R_COEFFICIENTS_BOX = True  # R^2, R^4, R^6, Sum, Dのボックスを表示するか
is_show_contour_value = False    # コンター値ラベルを表示するか

# ------------------------------------------------------------------
# 4. 電場範囲設定
# ------------------------------------------------------------------
E_range_abs = 400.0  # 電場範囲の絶対値（最大値は正、最小値は負になる）
E_divisions = 30 if SPEED_MODE else 60  # 電場範囲を何分割するか（スピードモード: 25, 品質モード: 60）

# ------------------------------------------------------------------
# 5. 格子・描画範囲設定
# ------------------------------------------------------------------
# Pの範囲の基準値（単位変換を適用しない場合の値）
P_range_abs_base = 30
# 単位変換を適用する場合、Pの範囲を1/100にする（変換後の値が元の範囲内に収まるように）
P_range_abs = P_range_abs_base / 100.0 if APPLY_UNIT_CONVERSION else P_range_abs_base
P_range = (-P_range_abs, P_range_abs)  # P軸の範囲（格子・描画共通）
# Rの範囲の基準値（単位変換を適用しない場合の値）
R_range_base = (0, 7)
# 単位変換を適用する場合、Rの範囲を1/5倍にする（変換後のrの範囲が変換前のRの範囲と同じくらいになるように）
# 変換式: r = l*φ/2 = 0.2*R より、R = r/0.2 = 5*r
# 変換前のRの範囲を(0, 1.4)にすると、変換後のrの範囲は(0, 7)となり、変換前のRの範囲(0, 7)と同じくらいになる
R_range = (R_range_base[0] / 5.0, R_range_base[1] / 5.0) if APPLY_UNIT_CONVERSION else R_range_base
# メッシュサイズの基準値（単位変換を適用しない場合の値）
grid_mesh_base = 0.01
# 単位変換を適用する場合、メッシュサイズを1/10倍にする
grid_mesh = grid_mesh_base / 10.0 if APPLY_UNIT_CONVERSION else grid_mesh_base

# ------------------------------------------------------------------
# 6. エネルギー・カットオフ設定
# ------------------------------------------------------------------
ENERGY_CUTOFF = 5000.0  # エネルギーカットオフ値

# ------------------------------------------------------------------
# 7. 保存先ディレクトリ設定
# ------------------------------------------------------------------
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "Outputs_E")
# 描画モードとスピードモードでフォルダ名を変更
# シンプル描画モードの場合は、temp_E_speed/temp_E_qualityの中にsimple_modeディレクトリを作成
# シンプル描画モードがTrueの時はシンプル描画モードの出力のみ、Falseの時は通常モードのみの出力
if SPEED_MODE:
    base_dir = os.path.join(OUTPUTS_DIR, "temp_E_speed")
else:
    base_dir = os.path.join(OUTPUTS_DIR, "temp_E_quality")

if SIMPLE_PLOT_MODE:
    GRAPH_SAVE_DIR = os.path.join(base_dir, "simple_mode")
else:
    GRAPH_SAVE_DIR = base_dir

# ==================================================================
# 【自動計算される設定値（変更不要）】
# ==================================================================

# ------------------------------------------------------------------
# 初期値設定
# ------------------------------------------------------------------
init_P_value = 0.0  # Pの初期値
init_R_value = 0.0  # Rの初期値

# ------------------------------------------------------------------
# 動作制御フラグ（通常は変更不要）
# ------------------------------------------------------------------
SKIP_FIRST_CYCLE_PLOT = True    # 1周目をスキップしてプロットしない
SHOW_TRAJECTORY = not SPEED_MODE  # 軌跡を表示するか（品質優先モードのみ）
AUTO_DELETE_PNG_AFTER_GIF = True  # GIF生成後にPNGを自動削除するか（全モード共通）

# 電場範囲の最大値・最小値
E_max = E_range_abs
E_min = -E_range_abs

# 後方互換性のための設定（P_range, R_rangeから自動設定）
P_grid = P_range
R_grid = R_range
contour_graph_x = P_range
contour_graph_y = R_range
P_E_graph_y = P_range
R_E_graph_y = R_range

# 増加→減少→増加のヒステリシスを考慮した電場範囲（1回目）
E_range_1st = np.concatenate([
    np.linspace(0, E_max, E_divisions + 1)[:-1],  # 0からE_maxまで（E_maxは含まない）
    np.linspace(E_max, E_min, E_divisions + 1)[:-1],  # E_maxからE_minまで（E_minは含まない）
    np.linspace(E_min, 0, E_divisions + 1)  # E_minから0まで（0は含む）
])
 
# 2回目の電場印加（1回目と同じパターン）
E_range_2nd = np.concatenate([
    np.linspace(0, E_max, E_divisions + 1)[:-1],  # 0からE_maxまで（E_maxは含まない）
    np.linspace(E_max, E_min, E_divisions + 1)[:-1],  # E_maxからE_minまで（E_minは含まない）
    np.linspace(E_min, 0, E_divisions + 1)  # E_minから0まで（0は含む）
])

# 全体の電場範囲（1回目 + 2回目）
E_range = np.concatenate([E_range_1st, E_range_2nd])

# ==================================================================
# 【関数定義】
# ==================================================================

# ------------------------------------------------------------------
def format_value_with_color(value):
    """有効数字4桁で表示する関数"""
    if value == 0:
        return "0.000"
    
    abs_value = abs(value)
    # 有効数字4桁を保証するため、まず科学的記法で表示してから
    # 値の大きさに応じて通常表記に変換
    if abs_value >= 0.0001 and abs_value < 10000:
        # 0.0001以上10000未満: 通常表記で有効数字4桁
        # 小数点以下の桁数を適切に計算
        if abs_value >= 1:
            # 整数部分の桁数
            int_digits = int(math.log10(abs_value)) + 1
            dec_places = max(0, 4 - int_digits)
            formatted = f"{value:.{dec_places}f}".rstrip('0').rstrip('.')
        else:
            # 1未満: 最初のゼロでない桁までの桁数
            dec_places = int(-math.log10(abs_value)) + 4
            formatted = f"{value:.{dec_places}f}".rstrip('0').rstrip('.')
    else:
        # 科学的記法で有効数字4桁（小数点以下3桁）
        formatted = f"{value:.3e}"
    
    if value < 0:
        return f"$\\mathbf{{{formatted}}}$"  # 太字でマイナス値を強調
    else:
        return formatted

# ------------------------------------------------------------------
def create_r_coefficients_annotation(R2_val, R4_val, R6_val, total_val, discriminant_D):
    """R^2, R^4, R^6, Sum, Dのアノテーションテキストを生成する共通関数"""
    return (
        f"$R^2$={format_value_with_color(R2_val)}\n"
        f"$R^4$={format_value_with_color(R4_val)}\n"
        f"$R^6$={format_value_with_color(R6_val)}\n"
        f"Sum={format_value_with_color(total_val)}\n"
        f"$D$={format_value_with_color(discriminant_D)}"
    )

# ------------------------------------------------------------------
def calc_r_coefficients(P_at_stable, params):
    """R^2, R^4, R^6の係数値を計算する関数（重複計算を避けるため、Pの単位: [C/m^2]）"""
    T = params['T_FIXED']
    beta_1_eff = params['beta_1'] * (T - params['t_{cR}'])
    P2 = P_at_stable ** 2
    P4 = P2 ** 2
    P6 = P4 * P2
    P8 = P4 ** 2
    
    # 単位変換係数の計算
    if APPLY_UNIT_CONVERSION:
        # Pの単位変換: [uC/cm^2] → [C/m^2] のため、各P項の係数を100^(P次数)倍
        # Rの単位変換: 角度定義R（φ）→ 変位定義r = l*φ/2 のため、各R項の係数を(2/l)^(R次数)倍
        # R = r/(l/2) = r/0.2 = 5*r より、R^2 → 25*r^2, R^4 → 625*r^4, R^6 → 15625*r^6
        P_conv_2 = 100**2  # P^2項: 10000
        P_conv_4 = 100**4  # P^4項: 100000000
        P_conv_6 = 100**6  # P^6項: 1000000000000
        P_conv_8 = 100**8  # P^8項: 10000000000000000
        R_conv_2 = (2.0/LATTICE_CONSTANT_L)**2  # R^2項: 25
        R_conv_4 = (2.0/LATTICE_CONSTANT_L)**4  # R^4項: 625
        R_conv_6 = (2.0/LATTICE_CONSTANT_L)**6  # R^6項: 15625
    else:
        # 単位変換を適用しない場合、係数は1
        P_conv_2 = P_conv_4 = P_conv_6 = P_conv_8 = 1.0
        R_conv_2 = R_conv_4 = R_conv_6 = 1.0
    
    R2_val = (
        0.5 * beta_1_eff * R_conv_2  # R^2項
        + params['gamma_22'] * P2 * P_conv_2 * R_conv_2  # P^2*R^2項
        + params['gamma_42'] * P4 * P_conv_4 * R_conv_2  # P^4*R^2項
        + params['gamma_82'] * P8 * P_conv_8 * R_conv_2  # P^8*R^2項
    ) 
    R4_val = (
        0.25 * params['beta_2'] * R_conv_4  # R^4項
        + params['gamma_64'] * P6 * P_conv_6 * R_conv_4  # P^6*R^4項
        + params['gamma_84'] * P8 * P_conv_8 * R_conv_4  # P^8*R^4項
    )
    R6_val = (1.0/6.0) * params['beta_3'] * R_conv_6 + params['gamma_86'] * P8 * P_conv_8 * R_conv_6  # R^6項, P^8*R^6項
    total_val = R2_val + R4_val + R6_val
    
    # 判別式D = c²(b²-4ac) を計算（R6_val=a, R4_val=b, R2_val=c）
    a = R6_val
    b = R4_val
    c = R2_val
    discriminant_D = c**2 * (b**2 - 4*a*c)
    
    return R2_val, R4_val, R6_val, total_val, discriminant_D

# ------------------------------------------------------------------
def calc_free_energy(P, R, e, params):
    """自由エネルギーを計算（Pの単位: [C/m^2]）"""
    T = params['T_FIXED'] # 固定温度をパラメータから取得
    alpha_1_t = params['alpha_1'] * (T - params['t_{cP}'])
    beta_1_eff = params['beta_1'] * (T - params['t_{cR}'])
    # べき乗計算を最適化（P**2, P**4, P**6, P**8を一度計算して再利用）
    P2 = P**2
    P4 = P2**2
    P6 = P4 * P2
    P8 = P4**2
    R2 = R**2
    R4 = R2**2
    R6 = R4 * R2
    # 単位変換係数の計算
    if APPLY_UNIT_CONVERSION:
        # Pの単位変換: [uC/cm^2] → [C/m^2] のため、各P項の係数を100^(P次数)倍
        # Rの単位変換: 角度定義R（φ）→ 変位定義r = l*φ/2 のため、各R項の係数を(2/l)^(R次数)倍
        # R = r/(l/2) = r/0.2 = 5*r より、R^2 → 25*r^2, R^4 → 625*r^4, R^6 → 15625*r^6
        P_conv_1 = 100**1  # P^1項（電場項）: 100
        P_conv_2 = 100**2  # P^2項: 10000
        P_conv_4 = 100**4  # P^4項: 100000000
        P_conv_6 = 100**6  # P^6項: 1000000000000
        P_conv_8 = 100**8  # P^8項: 10000000000000000
        R_conv_2 = (2.0/LATTICE_CONSTANT_L)**2  # R^2項: 25
        R_conv_4 = (2.0/LATTICE_CONSTANT_L)**4  # R^4項: 625
        R_conv_6 = (2.0/LATTICE_CONSTANT_L)**6  # R^6項: 15625
    else:
        # 単位変換を適用しない場合、係数は1
        P_conv_1 = P_conv_2 = P_conv_4 = P_conv_6 = P_conv_8 = 1.0
        R_conv_2 = R_conv_4 = R_conv_6 = 1.0
    
    return (
        0.5 * alpha_1_t * P2 * P_conv_2  # P^2項
        + 0.25 * params['alpha_2'] * P4 * P_conv_4  # P^4項
        + (1.0/6.0) * params['alpha_3'] * P6 * P_conv_6  # P^6項
        + 0.5 * beta_1_eff * R2 * R_conv_2  # R^2項
        + 0.25 * params['beta_2'] * R4 * R_conv_4  # R^4項
        + (1.0/6.0) * params['beta_3'] * R6 * R_conv_6  # R^6項
        + params['gamma_22'] * P2 * R2 * P_conv_2 * R_conv_2  # P^2*R^2項
        + params['gamma_42'] * P4 * R2 * P_conv_4 * R_conv_2  # P^4*R^2項
        + params['gamma_82'] * P8 * R2 * P_conv_8 * R_conv_2  # P^8*R^2項
        + params['gamma_64'] * P6 * R4 * P_conv_6 * R_conv_4  # P^6*R^4項
        + params['gamma_84'] * P8 * R4 * P_conv_8 * R_conv_4  # P^8*R^4項
        + params['gamma_86'] * P8 * R6 * P_conv_8 * R_conv_6  # P^8*R^6項
        - P * e * P_conv_1  # P^1項（電場項）
    )

# 近傍点のリストを定数として定義（毎回作成するのを避ける）
NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1), ( 0, 0), ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
]

def find_stable_point(P, R, free_energy, init_p, init_r):
    """安定点探索"""
    idx_p = (np.abs(P[:,0] - init_p)).argmin()
    idx_r = (np.abs(R[0,:] - init_r)).argmin()
    shift_point = (idx_p, idx_r)

    for _ in range(10000):
        center_p, center_r = shift_point
        min_point = shift_point
        min_energy = free_energy[center_p, center_r]

        for dP, dR in NEIGHBORS:
            new_p = center_p + dP
            new_r = center_r + dR
            if (0 <= new_p < P.shape[0]) and (0 <= new_r < R.shape[1]):
                if free_energy[new_p, new_r] < min_energy:
                    min_energy = free_energy[new_p, new_r]
                    min_point = (new_p, new_r)

        if min_point == shift_point:
            break
        else:
            shift_point = min_point

    stable_idx_p, stable_idx_r = shift_point 
    return P[stable_idx_p, stable_idx_r], R[stable_idx_p, stable_idx_r]

def plot_2d_contour(P, R, free_energy, stable_p, stable_r, e, n, subdir, params, z_lim=None, param_text="", prev_stable_p=None, prev_stable_r=None, simple_mode=None):
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # エネルギーカットオフでマスク
    free_energy_masked = np.ma.masked_where(free_energy >= ENERGY_CUTOFF, free_energy)
    
    if z_lim:
        # 低エネルギー領域を強調するため、より細かいコンターを描画
        levels = np.linspace(z_lim[0], z_lim[1], 80 if SPEED_MODE else 50)
    else:
        # マスクされた値以外の最小値・最大値でレベルを設定
        valid_energy = free_energy_masked[~free_energy_masked.mask] if np.any(~free_energy_masked.mask) else free_energy
        levels = np.linspace(np.min(valid_energy), np.max(valid_energy), 80 if SPEED_MODE else 50)

    # 低エネルギー領域の勾配をみやすくするため、より細かいコンターを描画
    contour = ax.contour(P, R, free_energy_masked, levels=levels, cmap="jet", linewidths=0.8 if SPEED_MODE else 1.0)
    if is_show_contour_value:
        # 低エネルギー領域のコンターラベルをより多く表示
        step = 3 if SPEED_MODE else 5
        contour.clabel(contour.levels[::step], fmt='%.02f' if not SPEED_MODE else '%.01f', fontsize=5 if SPEED_MODE else 6)

    # 前の安定点から現在の安定点への軌跡を描画（SHOW_TRAJECTORYがTrueの場合、シンプルモードでも表示）
    if SHOW_TRAJECTORY and prev_stable_p is not None and prev_stable_r is not None:
        if not simple_mode:
            ax.plot([prev_stable_p, stable_p], [prev_stable_r, stable_r], 'r-', linewidth=4, alpha=0.6, label='Trajectory')
        else:
            ax.plot([prev_stable_p, stable_p], [prev_stable_r, stable_r], 'r-', linewidth=4, alpha=0.6)
        ax.plot(prev_stable_p, prev_stable_r, 'ro', markersize=6, alpha=0.2)  # 前の点を薄い赤で表示
    
    ax.plot(stable_p, stable_r, 'ro', markersize=8)
    # 安定点のPに基づく R^2, R^4, R^6 の係数値を計算して枠外に注記
    R2_val, R4_val, R6_val, total_val, discriminant_D = calc_r_coefficients(stable_p, params)
    
    if not simple_mode:
        ax.set_title(f"2D Free Energy Map at E={e:.3f}")
    ax.set_xlabel("P'" if APPLY_UNIT_CONVERSION else "P")
    ax.set_ylabel("R'" if APPLY_UNIT_CONVERSION else "R")
    ax.set_xlim(contour_graph_x)
    ax.set_ylim(contour_graph_y)
    ax.grid()  # シンプルモードでもグリッドを表示
    if not simple_mode:
        if SHOW_TRAJECTORY and prev_stable_p is not None:
            ax.legend()

        ax.text(1.05, 0.95, param_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
        
        # R^2, R^4, R^6, Sum, Dのボックスを表示するかチェック
        if SHOW_R_COEFFICIENTS_BOX:
            annotation_text = create_r_coefficients_annotation(R2_val, R4_val, R6_val, total_val, discriminant_D)
            # パラメータボックスの下に適切な間隔を空けて変数値ボックスを配置
            # パラメータは11行なので、その高さを考慮して十分下に配置
            ax.text(1.05, 0.30, annotation_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # ディレクトリはメインループで既に作成済み
    plt.savefig(os.path.join(subdir, f"2Dcontour_{n:04d}.png"), bbox_inches='tight')
    plt.close()

def plot_3d_surface(P, R, free_energy, stable_p, stable_r, e, n, subdir, params, z_lim=None, param_text="", simple_mode=None):
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    fig = plt.figure(figsize=(12 if SPEED_MODE else 8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # エネルギーカットオフでクリップ
    free_energy_smooth = np.where(free_energy >= ENERGY_CUTOFF, ENERGY_CUTOFF, free_energy)
    
    ax.plot_surface(P, R, free_energy_smooth, cmap=cm.jet, linewidth=0, antialiased=False, alpha=0.6)
    stable_energy = free_energy[(np.abs(P[:,0]-stable_p)).argmin(), (np.abs(R[0,:]-stable_r)).argmin()]
    ax.scatter(stable_p, stable_r, stable_energy, color='r', s=50)
    # 安定点のPに基づく R^2, R^4, R^6 の係数値を計算して枠外に注記
    R2_val, R4_val, R6_val, total_val, discriminant_D = calc_r_coefficients(stable_p, params)
    
    if not simple_mode:
        ax.set_title(f"3D Free Energy Map at E={e:.3f}", pad=2)
    ax.set_xlabel("P'" if APPLY_UNIT_CONVERSION else "P")
    ax.set_ylabel("R'" if APPLY_UNIT_CONVERSION else "R")
    zlabel = "F(P',R')" if APPLY_UNIT_CONVERSION else "F(P,R)"
    ax.set_zlabel(zlabel, labelpad=10)  # ラベルの位置を調整して見切れを防ぐ

    if z_lim:
        ax.set_zlim(z_lim)

    ax.view_init(elev=20, azim=-60)

    if not simple_mode:
        ax.text2D(1.15, 0.95, param_text, transform=ax.transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
        
        # R^2, R^4, R^6, Sum, Dのボックスを表示するかチェック
        if SHOW_R_COEFFICIENTS_BOX:
            annotation_text = create_r_coefficients_annotation(R2_val, R4_val, R6_val, total_val, discriminant_D)
            # パラメータボックスの下に適切な間隔を空けて変数値ボックスを配置
            # パラメータは11行なので、その高さを考慮して十分下に配置
            # 軸表示と重ならないよう、さらに右に配置
            ax.text2D(1.15, 0.30, annotation_text, transform=ax.transAxes,
                      fontsize=9, verticalalignment='top', horizontalalignment='left',
                      bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    # ディレクトリはメインループで既に作成済み
    plt.savefig(os.path.join(subdir, f"3Dsurf_{n:04d}.png"), bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_cross_section_P(P, R, free_energy, stable_p, stable_r, e, n, subdir, params, z_lim=None, param_text="", simple_mode=None):
    """P軸に垂直な断面でのエネルギー最下点をプロット"""
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # エネルギー最下点を各P値で探索（ベクトル化）
    # マスクされた値や極端に大きな値を除外
    energy_masked = np.where(np.isnan(free_energy) | (free_energy >= ENERGY_CUTOFF), np.inf, free_energy)
    min_energies = np.min(energy_masked, axis=1)
    # 全てinfの場合はnanに変換
    min_energies = np.where(np.isinf(min_energies), np.nan, min_energies)
    
    P_values = P[:, 0]
    
    # プロット
    if not simple_mode:
        ax.plot(P_values, min_energies, 'b-', linewidth=1.5, label='Energy minimum')
    else:
        ax.plot(P_values, min_energies, 'b-', linewidth=1.5)
    
    # stable_pに対応するエネルギー値を探す
    stable_idx = (np.abs(P_values - stable_p)).argmin()
    stable_energy = min_energies[stable_idx]
    if not simple_mode:
        p_label = "P'" if APPLY_UNIT_CONVERSION else "P"
        ax.plot(stable_p, stable_energy, 'ro', markersize=10, label=f'Stable {p_label}={stable_p:.3f}')
    else:
        ax.plot(stable_p, stable_energy, 'ro', markersize=10)
    
    # 安定点のPに基づく R^2, R^4, R^6 の係数値を計算して枠外に注記
    R2_val, R4_val, R6_val, total_val, discriminant_D = calc_r_coefficients(stable_p, params)
    
    if not simple_mode:
        ax.set_title(f"Cross-section Energy Min (R-axis) at E={e:.3f}")
    ax.set_xlabel("P'" if APPLY_UNIT_CONVERSION else "P")
    ax.set_ylabel("Minimum Energy")
    ax.set_xlim(contour_graph_x)
    if z_lim:
        ax.set_ylim(z_lim)
    ax.grid(True, alpha=0.3)  # シンプルモードでもグリッドを表示
    if not simple_mode:
        ax.legend()
        
        ax.text(1.05, 0.95, param_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
        
        # R^2, R^4, R^6, Sum, Dのボックスを表示するかチェック
        if SHOW_R_COEFFICIENTS_BOX:
            annotation_text = create_r_coefficients_annotation(R2_val, R4_val, R6_val, total_val, discriminant_D)
            # パラメータボックスの下に適切な間隔を空けて変数値ボックスを配置
            # パラメータは11行なので、その高さを考慮して十分下に配置
            ax.text(1.05, 0.30, annotation_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # ディレクトリはメインループで既に作成済み
    plt.savefig(os.path.join(subdir, f"cross_section_P_{n:04d}.png"), bbox_inches='tight')
    plt.close()

def plot_cross_section_R(P, R, free_energy, stable_p, stable_r, e, n, subdir, params, z_lim=None, param_text="", simple_mode=None):
    """R軸に垂直な断面でのエネルギー最下点をプロット"""
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # エネルギー最下点を各R値で探索（ベクトル化）
    # マスクされた値や極端に大きな値を除外
    energy_masked = np.where(np.isnan(free_energy) | (free_energy >= ENERGY_CUTOFF), np.inf, free_energy)
    min_energies = np.min(energy_masked, axis=0)
    # 全てinfの場合はnanに変換
    min_energies = np.where(np.isinf(min_energies), np.nan, min_energies)
    
    R_values = R[0, :]
    
    # プロット
    if not simple_mode:
        ax.plot(R_values, min_energies, 'b-', linewidth=1.5, label='Energy minimum')
    else:
        ax.plot(R_values, min_energies, 'b-', linewidth=1.5)
    
    # stable_rに対応するエネルギー値を探す
    stable_idx = (np.abs(R_values - stable_r)).argmin()
    stable_energy = min_energies[stable_idx]
    if not simple_mode:
        r_label = "R'" if APPLY_UNIT_CONVERSION else "R"
        ax.plot(stable_r, stable_energy, 'ro', markersize=10, label=f'Stable {r_label}={stable_r:.3f}')
    else:
        ax.plot(stable_r, stable_energy, 'ro', markersize=10)
    
    # 安定点のPに基づく R^2, R^4, R^6 の係数値を計算して枠外に注記
    R2_val, R4_val, R6_val, total_val, discriminant_D = calc_r_coefficients(stable_p, params)
    
    if not simple_mode:
        ax.set_title(f"Cross-section Energy Min (P-axis) at E={e:.3f}")
    ax.set_xlabel("R'" if APPLY_UNIT_CONVERSION else "R")
    ax.set_ylabel("Minimum Energy")
    ax.set_xlim(contour_graph_y)
    if z_lim:
        ax.set_ylim(z_lim)
    ax.grid(True, alpha=0.3)  # シンプルモードでもグリッドを表示
    if not simple_mode:
        ax.legend()
        
        ax.text(1.05, 0.95, param_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
        
        # R^2, R^4, R^6, Sum, Dのボックスを表示するかチェック
        if SHOW_R_COEFFICIENTS_BOX:
            annotation_text = create_r_coefficients_annotation(R2_val, R4_val, R6_val, total_val, discriminant_D)
            # パラメータボックスの下に適切な間隔を空けて変数値ボックスを配置
            # パラメータは11行なので、その高さを考慮して十分下に配置
            ax.text(1.05, 0.30, annotation_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # ディレクトリはメインループで既に作成済み
    plt.savefig(os.path.join(subdir, f"cross_section_R_{n:04d}.png"), bbox_inches='tight')
    plt.close()

def plot_hysteresis_P_E_frame(E_array, P_stable_array, E_current, P_current, current_idx, n, subdir, params, param_text="", simple_mode=None):
    """P-Eヒステリシスループの各フレームを生成（現在の点をハイライト）"""
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ヒステリシスループ全体を描画（通常色で不透明度90%、見やすくする）
    if not simple_mode:
        ax.plot(E_array, P_stable_array, 'o-', markersize=2, linewidth=1, 
                color='blue', alpha=0.9, label="P(E) loop")
        
        # 現在の点までを強調表示（不透明度100%）
        if current_idx >= 0 and current_idx < len(E_array):
            ax.plot(E_array[:current_idx+1], P_stable_array[:current_idx+1], 
                    'o-', markersize=3, linewidth=2, color='blue', alpha=1.0, label="P(E) path")
        
        # 現在の点をハイライト（大きく赤い点）
        ax.plot(E_current, P_current, 'ro', markersize=12, label=f'Current E={E_current:.3f}')
        ax.plot(E_current, P_current, 'ro', markersize=8, markerfacecolor='none', markeredgewidth=2)
    else:
        ax.plot(E_array, P_stable_array, 'o-', markersize=2, linewidth=1, 
                color='blue', alpha=0.9)
        
        # 現在の点までを強調表示（不透明度100%）
        if current_idx >= 0 and current_idx < len(E_array):
            ax.plot(E_array[:current_idx+1], P_stable_array[:current_idx+1], 
                    'o-', markersize=3, linewidth=2, color='blue', alpha=1.0)
        
        # 現在の点をハイライト（大きく赤い点）
        ax.plot(E_current, P_current, 'ro', markersize=12)
        ax.plot(E_current, P_current, 'ro', markersize=8, markerfacecolor='none', markeredgewidth=2)
    
    if not simple_mode:
        ax.set_title(f"P vs Electric Field (Hysteresis Loop) at E={E_current:.3f}")
    ax.set_xlabel("E")
    ax.set_ylabel("P'" if APPLY_UNIT_CONVERSION else "P")
    ax.set_ylim(P_E_graph_y)
    ax.grid(True, alpha=0.3)  # シンプルモードでもグリッドを表示
    if not simple_mode:
        ax.legend(loc='best', fontsize=8)
        
        ax.text(1.05, 0.95, param_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
    
    # ディレクトリはメインループで既に作成済み
    plt.savefig(os.path.join(subdir, f"hysteresis_P_E_{n:04d}.png"), bbox_inches='tight')
    plt.close()

def plot_hysteresis_R_E_frame(E_array, R_stable_array, E_current, R_current, current_idx, n, subdir, params, param_text="", simple_mode=None):
    """R-Eヒステリシスループの各フレームを生成（現在の点をハイライト）"""
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ヒステリシスループ全体を描画（通常色で不透明度90%、見やすくする）
    if not simple_mode:
        ax.plot(E_array, R_stable_array, 'o-', markersize=2, linewidth=1, 
                color='green', alpha=0.9, label="R(E) loop")
        
        # 現在の点までを強調表示（不透明度100%）
        if current_idx >= 0 and current_idx < len(E_array):
            ax.plot(E_array[:current_idx+1], R_stable_array[:current_idx+1], 
                    'o-', markersize=3, linewidth=2, color='green', alpha=1.0, label="R(E) path")
        
        # 現在の点をハイライト（大きく赤い点）
        ax.plot(E_current, R_current, 'ro', markersize=12, label=f'Current E={E_current:.3f}')
        ax.plot(E_current, R_current, 'ro', markersize=8, markerfacecolor='none', markeredgewidth=2)
    else:
        ax.plot(E_array, R_stable_array, 'o-', markersize=2, linewidth=1, 
                color='green', alpha=0.9)
        
        # 現在の点までを強調表示（不透明度100%）
        if current_idx >= 0 and current_idx < len(E_array):
            ax.plot(E_array[:current_idx+1], R_stable_array[:current_idx+1], 
                    'o-', markersize=3, linewidth=2, color='green', alpha=1.0)
        
        # 現在の点をハイライト（大きく赤い点）
        ax.plot(E_current, R_current, 'ro', markersize=12)
        ax.plot(E_current, R_current, 'ro', markersize=8, markerfacecolor='none', markeredgewidth=2)
    
    if not simple_mode:
        ax.set_title(f"R vs Electric Field (Hysteresis Loop) at E={E_current:.3f}")
    ax.set_xlabel("E")
    ax.set_ylabel("R'" if APPLY_UNIT_CONVERSION else "R")
    ax.set_ylim(R_E_graph_y)
    ax.grid(True, alpha=0.3)  # シンプルモードでもグリッドを表示
    if not simple_mode:
        ax.legend(loc='best', fontsize=8)
        
        ax.text(1.05, 0.95, param_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
    
    # ディレクトリはメインループで既に作成済み
    plt.savefig(os.path.join(subdir, f"hysteresis_R_E_{n:04d}.png"), bbox_inches='tight')
    plt.close()

def get_perovskite_structure(nx=1, ny=1, nz=1, a=1.0):
    """
    BaTiO3ペロブスカイト構造の原子位置を生成
    
    Parameters:
    -----------
    nx, ny, nz : int
        各軸方向のユニットセル数
    a : float
        格子定数（正規化された単位）
    
    Returns:
    --------
    ba_positions : array
        Ba原子（Aサイト）の位置 [N, 3]
    ti_positions : array
        Ti原子（Bサイト）の位置 [N, 3]
    o_positions : array
        O原子（Xサイト）の位置 [N, 3]
    octahedra_info : list
        各八面体の情報 [(ti_idx, [o_indices]), ...]
    """
    ba_positions = []
    ti_positions = []
    o_positions_dict = {}  # 位置をキーとしてO原子を管理（重複回避）
    octahedra_info = []  # [(center_idx, [vertex_indices]), ...]
    
    def get_o_index(pos, tolerance=1e-6):
        """O原子の位置からインデックスを取得（重複チェック）"""
        pos_key = tuple(np.round(pos / tolerance) * tolerance)
        if pos_key not in o_positions_dict:
            o_positions_dict[pos_key] = len(o_positions_dict)
        return o_positions_dict[pos_key]
    
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # ユニットセルの原点
                base = np.array([ix * a, iy * a, iz * a])
                
                # Ba原子（Aサイト）：立方体の8つの角（重複あり）
                for dz in [0, 1]:
                    for dy in [0, 1]:
                        for dx in [0, 1]:
                            ba_positions.append(base + np.array([dx * a, dy * a, dz * a]))
                
                # Ti原子（Bサイト）：立方体の中心
                ti_center = base + np.array([0.5 * a, 0.5 * a, 0.5 * a])
                ti_idx = len(ti_positions)
                ti_positions.append(ti_center)
                
                # O原子（Xサイト）：立方体の6つの面の中心
                # このTi原子に属するO原子のインデックス
                o_indices = []
                
                # 面心のO原子（6つ）- 重複を避けるため、位置ベースで管理
                face_centers = [
                    base + np.array([0.5 * a, 0.5 * a, 0.0 * a]),  # 底面
                    base + np.array([0.5 * a, 0.5 * a, 1.0 * a]),  # 上面
                    base + np.array([0.5 * a, 0.0 * a, 0.5 * a]),  # 前面
                    base + np.array([0.5 * a, 1.0 * a, 0.5 * a]),  # 後面
                    base + np.array([0.0 * a, 0.5 * a, 0.5 * a]),  # 左面
                    base + np.array([1.0 * a, 0.5 * a, 0.5 * a]),   # 右面
                ]
                
                for face_pos in face_centers:
                    o_idx = get_o_index(face_pos)
                    o_indices.append(o_idx)
                
                octahedra_info.append((ti_idx, o_indices))
    
    # O原子の位置リストを作成（ソート済み）
    o_positions_list = sorted(o_positions_dict.items(), key=lambda x: x[1])
    o_positions = np.array([np.array(pos) for pos, _ in o_positions_list])
    
    # octahedra_infoのインデックスを更新（O原子の順序に合わせる）
    o_pos_to_new_idx = {pos: idx for idx, (pos, _) in enumerate(o_positions_list)}
    
    octahedra_info_updated = []
    for ti_idx, o_indices_old in octahedra_info:
        # 元のインデックスから位置を取得して、新しいインデックスに変換
        old_positions = list(o_positions_dict.keys())
        o_indices_new = [o_pos_to_new_idx[old_positions[old_idx]] 
                         for old_idx in o_indices_old if old_idx < len(old_positions)]
        octahedra_info_updated.append((ti_idx, o_indices_new))
    
    ba_positions = np.array(ba_positions)
    ti_positions = np.array(ti_positions)
    
    return ba_positions, ti_positions, o_positions, octahedra_info_updated

def apply_polarization_and_rotation(ba_positions, ti_positions, o_positions, octahedra_info, 
                                     P, R, P_max, R_max, a=1.0):
    """
    P（分極）とR（八面体回転）を構造に適用
    
    Parameters:
    -----------
    ba_positions, ti_positions, o_positions : array
        原子位置
    octahedra_info : list
        八面体情報
    P : float
        分極値（正規化）
    R : float
        八面体回転角度（度、正規化）
    P_max : float
        Pの最大値（スケーリング用）
    R_max : float
        Rの最大値（スケーリング用）
    a : float
        格子定数
    
    Returns:
    --------
    ti_positions_new : array
        変形後のTi原子位置
    o_positions_new : array
        変形後のO原子位置
    """
    # スケーリング：最大値で正規化して、破綻を防ぐ
    P_scaled = P / max(abs(P_max), 1e-10) if P_max != 0 else 0
    R_scaled = R / max(abs(R_max), 1e-10) if R_max != 0 else 0
    
    # Pに基づくTi原子の変位（c軸方向に分極）
    # 最大変位は格子定数の10%程度に制限
    max_displacement = 0.1 * a
    ti_displacement = np.array([0, 0, P_scaled * max_displacement])
    ti_positions_new = ti_positions.copy() + ti_displacement
    
    # Rに基づく酸素八面体の回転（c軸周り）
    # 回転角度を度からラジアンに変換（最大15度程度に制限）
    max_rotation_deg = 15.0
    rotation_angle = np.deg2rad(R_scaled * max_rotation_deg)
    
    o_positions_new = o_positions.copy()
    
    # Rが非常に小さい場合（回転しない場合）は計算をスキップ
    if abs(rotation_angle) > 1e-6:
        # 各O原子がどの八面体に属しているかを記録（重複を考慮）
        o_to_octahedra = {}
        for ti_idx, o_indices in octahedra_info:
            for o_idx in o_indices:
                if o_idx not in o_to_octahedra:
                    o_to_octahedra[o_idx] = []
                o_to_octahedra[o_idx].append(ti_idx)
        
        # 各O原子に対して、それが属する全ての八面体の回転を考慮して変位を計算
        for o_idx in range(len(o_positions)):
            o_pos_original = o_positions[o_idx]
            
            # このO原子が属する全ての八面体の中心からの変位を計算
            displacements = []
            
            if o_idx in o_to_octahedra:
                for ti_idx in o_to_octahedra[o_idx]:
                    ti_center = ti_positions_new[ti_idx]
                    
                    # Ti原子の位置からユニットセルのインデックスを逆算
                    ix = int(round((ti_center[0] - 0.5 * a) / a))
                    iy = int(round((ti_center[1] - 0.5 * a) / a))
                    
                    # チェッカーボードパターンで回転の向きを決定
                    rotation_sign = 1 if (ix + iy) % 2 == 0 else -1
                    actual_rotation_angle = rotation_angle * rotation_sign
                    
                    # Ti中心を原点とした相対座標
                    relative_pos = o_pos_original - ti_center
                    
                    # c軸（z軸）周りの回転（向きを考慮）
                    # 回転行列を使わずに直接計算（高速化）
                    cos_r = np.cos(actual_rotation_angle)
                    sin_r = np.sin(actual_rotation_angle)
                    # 2D回転を直接適用（z成分はそのまま）
                    x_rot = cos_r * relative_pos[0] - sin_r * relative_pos[1]
                    y_rot = sin_r * relative_pos[0] + cos_r * relative_pos[1]
                    rotated_relative = np.array([x_rot, y_rot, relative_pos[2]])
                    
                    # 絶対座標に変換
                    rotated_pos = ti_center + rotated_relative
                    displacements.append(rotated_pos)
            
            # 全ての八面体からの変位の平均を計算（一貫性を保つため）
            if displacements:
                o_positions_new[o_idx] = np.mean(displacements, axis=0)
            else:
                # 八面体に属していない場合は元の位置を保持
                o_positions_new[o_idx] = o_pos_original
    
    return ti_positions_new, o_positions_new

def plot_3d_unitcell_oblique(ba_positions, ti_positions, o_positions, octahedra_info,
                              P, R, E, n, subdir, param_text="", 
                              ti_positions_original=None, o_positions_original=None,
                              ti_positions_prev=None, o_positions_prev=None, simple_mode=None):
    """
    視点1：1ユニットセルを斜めから見た3Dモデル
    """
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 原子のサイズ（Aサイト、Bサイトを大きく）
    ba_size = 1400
    ti_size = 1000
    o_size = 400
    
    # ユニットセル全体を40%にスケール（原子のサイズはそのまま）
    scale_factor = 0.4
    ba_positions = ba_positions * scale_factor
    ti_positions = ti_positions * scale_factor
    o_positions = o_positions * scale_factor
    if ti_positions_original is not None:
        ti_positions_original = ti_positions_original * scale_factor
    if o_positions_original is not None:
        o_positions_original = o_positions_original * scale_factor
    if ti_positions_prev is not None:
        ti_positions_prev = ti_positions_prev * scale_factor
    if o_positions_prev is not None:
        o_positions_prev = o_positions_prev * scale_factor
    
    # 前のフレームの位置を残像として表示
    if ti_positions_prev is not None:
        ax.scatter(ti_positions_prev[:, 0], ti_positions_prev[:, 1], ti_positions_prev[:, 2],
                   s=ti_size, c='red', alpha=0.2, edgecolors='darkred', linewidths=0.5)
        # 前の位置から現在の位置まで線で結ぶ（グラデーション）
        if len(ti_positions_prev) == len(ti_positions):
            for i in range(len(ti_positions)):
                # 線を細かく分割してグラデーションを適用
                num_segments = 5
                x_segments = np.linspace(ti_positions_prev[i, 0], ti_positions[i, 0], num_segments + 1)
                y_segments = np.linspace(ti_positions_prev[i, 1], ti_positions[i, 1], num_segments + 1)
                z_segments = np.linspace(ti_positions_prev[i, 2], ti_positions[i, 2], num_segments + 1)
                # 現在位置側を濃く、前の位置側を薄くする
                for j in range(num_segments):
                    alpha = 0.6 * (j + 1) / num_segments  # 0から0.6までグラデーション
                    ax.plot([x_segments[j], x_segments[j+1]],
                           [y_segments[j], y_segments[j+1]],
                           [z_segments[j], z_segments[j+1]],
                           'r-', linewidth=12, alpha=alpha)
    
    if o_positions_prev is not None:
        ax.scatter(o_positions_prev[:, 0], o_positions_prev[:, 1], o_positions_prev[:, 2],
                   s=o_size, c='blue', alpha=0.2, edgecolors='darkblue', linewidths=0.3)
        # 前の位置から現在の位置まで線で結ぶ（グラデーション）
        if len(o_positions_prev) == len(o_positions):
            for i in range(len(o_positions)):
                # 線を細かく分割してグラデーションを適用
                num_segments = 5
                x_segments = np.linspace(o_positions_prev[i, 0], o_positions[i, 0], num_segments + 1)
                y_segments = np.linspace(o_positions_prev[i, 1], o_positions[i, 1], num_segments + 1)
                z_segments = np.linspace(o_positions_prev[i, 2], o_positions[i, 2], num_segments + 1)
                # 現在位置側を濃く、前の位置側を薄くする
                for j in range(num_segments):
                    alpha = 0.5 * (j + 1) / num_segments  # 0から0.5までグラデーション
                    ax.plot([x_segments[j], x_segments[j+1]],
                           [y_segments[j], y_segments[j+1]],
                           [z_segments[j], z_segments[j+1]],
                           'b-', linewidth=6, alpha=alpha)
    
    # 分極・回転が0の位置にBサイトとXサイトを薄く表示（結合線なし）
    if ti_positions_original is not None:
        ax.scatter(ti_positions_original[:, 0], ti_positions_original[:, 1], ti_positions_original[:, 2],
                   s=ti_size, c='red', alpha=0.15, edgecolors='darkred', linewidths=0.5)
    
    if o_positions_original is not None:
        ax.scatter(o_positions_original[:, 0], o_positions_original[:, 1], o_positions_original[:, 2],
                   s=o_size, c='blue', alpha=0.2, edgecolors='darkblue', linewidths=0.3)
    
    # Ba原子を描画（緑、透過表示）
    ax.scatter(ba_positions[:, 0], ba_positions[:, 1], ba_positions[:, 2],
               s=ba_size, c='green', alpha=0.3, label='Ba (A-site)', edgecolors='darkgreen', linewidths=1.5)
    
    # Ti原子を描画（赤、透過しない）
    ax.scatter(ti_positions[:, 0], ti_positions[:, 1], ti_positions[:, 2],
               s=ti_size, c='red', alpha=1.0, label='Ti (B-site)', edgecolors='darkred', linewidths=1.5)
    
    # 視点からの距離を計算して酸素原子を奥と手前に分類
    # 視点の設定（elev=15, azim=20）から方向ベクトルを計算
    elev_rad = np.deg2rad(15)
    azim_rad = np.deg2rad(20)
    view_direction = np.array([
        np.cos(elev_rad) * np.sin(azim_rad),
        np.cos(elev_rad) * np.cos(azim_rad),
        np.sin(elev_rad)
    ])
    
    # 各酸素原子の視点方向への射影（内積）を計算
    # 値が小さいほど奥（視点から遠い）、大きいほど手前（視点に近い）
    o_depths = np.dot(o_positions, view_direction)
    o_depth_median = np.median(o_depths)
    
    # 奥の酸素原子と手前の酸素原子に分類
    o_back_indices = np.where(o_depths < o_depth_median)[0]
    o_front_indices = np.where(o_depths >= o_depth_median)[0]
    
    # 奥の酸素原子を描画（八面体より先に描画）
    if len(o_back_indices) > 0:
        ax.scatter(o_positions[o_back_indices, 0], 
                   o_positions[o_back_indices, 1], 
                   o_positions[o_back_indices, 2],
                   s=o_size, c='blue', alpha=1.0, edgecolors='darkblue', linewidths=0.5)
    
    # 八面体を描画（半透明の青）- 奥のO原子と手前のO原子の間に描画
    for ti_idx, o_indices in octahedra_info:
        if ti_idx < len(ti_positions) and len(o_indices) >= 6:
            ti_center = ti_positions[ti_idx]
            o_vertices = o_positions[o_indices]
            
            # 正八面体の構造：Ti中心と6つのO原子
            # 6つのO原子は、Ti中心を中心として、±x, ±y, ±z方向に配置される
            # 8つの三角形面を定義：隣接する3つのO原子で構成される面
            
            # O原子をTi中心からの相対位置で分類
            # 各軸方向（x, y, z）に2つずつのO原子がある
            o_relative = o_vertices - ti_center
            
            # 各軸方向のO原子を特定
            o_x_pos = None
            o_x_neg = None
            o_y_pos = None
            o_y_neg = None
            o_z_pos = None
            o_z_neg = None
            
            for i, rel_pos in enumerate(o_relative):
                # 各軸方向の主要な成分を確認
                abs_rel = np.abs(rel_pos)
                max_idx = np.argmax(abs_rel)
                if max_idx == 0:  # x方向
                    if rel_pos[0] > 0:
                        o_x_pos = o_vertices[i]
                    else:
                        o_x_neg = o_vertices[i]
                elif max_idx == 1:  # y方向
                    if rel_pos[1] > 0:
                        o_y_pos = o_vertices[i]
                    else:
                        o_y_neg = o_vertices[i]
                elif max_idx == 2:  # z方向
                    if rel_pos[2] > 0:
                        o_z_pos = o_vertices[i]
                    else:
                        o_z_neg = o_vertices[i]
            
            # 8つの三角形面を定義（正八面体の構造）
            faces = []
            if o_x_pos is not None and o_y_pos is not None and o_z_pos is not None:
                faces.append([o_x_pos, o_y_pos, o_z_pos])
            if o_x_pos is not None and o_y_pos is not None and o_z_neg is not None:
                faces.append([o_x_pos, o_y_pos, o_z_neg])
            if o_x_pos is not None and o_y_neg is not None and o_z_pos is not None:
                faces.append([o_x_pos, o_y_neg, o_z_pos])
            if o_x_pos is not None and o_y_neg is not None and o_z_neg is not None:
                faces.append([o_x_pos, o_y_neg, o_z_neg])
            if o_x_neg is not None and o_y_pos is not None and o_z_pos is not None:
                faces.append([o_x_neg, o_y_pos, o_z_pos])
            if o_x_neg is not None and o_y_pos is not None and o_z_neg is not None:
                faces.append([o_x_neg, o_y_pos, o_z_neg])
            if o_x_neg is not None and o_y_neg is not None and o_z_pos is not None:
                faces.append([o_x_neg, o_y_neg, o_z_pos])
            if o_x_neg is not None and o_y_neg is not None and o_z_neg is not None:
                faces.append([o_x_neg, o_y_neg, o_z_neg])
            
            # Poly3DCollectionで描画（全ての面を着色）
            if len(faces) > 0:
                poly3d = Poly3DCollection(faces, alpha=0.2, facecolor='lightblue', 
                                         edgecolor='blue', linewidths=0.5)
                ax.add_collection3d(poly3d)
    
    # 手前の酸素原子を描画（青、透過しない）- 八面体の後に描画して手前の原子が半透明にならないようにする
    if len(o_front_indices) > 0:
        ax.scatter(o_positions[o_front_indices, 0], 
                   o_positions[o_front_indices, 1], 
                   o_positions[o_front_indices, 2],
                   s=o_size, c='blue', alpha=1.0, label='O (X-site)', edgecolors='darkblue', linewidths=0.5)
    
    # 結合線を描画（Ti-O結合）
    for ti_idx, o_indices in octahedra_info:
        if ti_idx < len(ti_positions):
            ti_center = ti_positions[ti_idx]
            for o_idx in o_indices:
                if o_idx < len(o_positions):
                    o_pos = o_positions[o_idx]
                    ax.plot([ti_center[0], o_pos[0]], 
                           [ti_center[1], o_pos[1]], 
                           [ti_center[2], o_pos[2]], 
                           'k-', linewidth=1, alpha=0.3)
    
    # O原子間の結合線を描画（八面体のエッジ）
    for ti_idx, o_indices in octahedra_info:
        if ti_idx < len(ti_positions) and len(o_indices) >= 6:
            ti_center = ti_positions[ti_idx]
            o_vertices = o_positions[o_indices]
            
            # O原子をTi中心からの相対位置で分類
            o_relative = o_vertices - ti_center
            
            # 各軸方向のO原子のインデックスを特定
            o_indices_by_axis = {'x_pos': None, 'x_neg': None, 
                                 'y_pos': None, 'y_neg': None,
                                 'z_pos': None, 'z_neg': None}
            
            for i, rel_pos in enumerate(o_relative):
                abs_rel = np.abs(rel_pos)
                max_idx = np.argmax(abs_rel)
                if max_idx == 0:  # x方向
                    if rel_pos[0] > 0:
                        o_indices_by_axis['x_pos'] = i
                    else:
                        o_indices_by_axis['x_neg'] = i
                elif max_idx == 1:  # y方向
                    if rel_pos[1] > 0:
                        o_indices_by_axis['y_pos'] = i
                    else:
                        o_indices_by_axis['y_neg'] = i
                elif max_idx == 2:  # z方向
                    if rel_pos[2] > 0:
                        o_indices_by_axis['z_pos'] = i
                    else:
                        o_indices_by_axis['z_neg'] = i
            
            # 隣接するO原子間の結合線を描画（正八面体のエッジ）
            # 直交する軸方向のO原子同士が接続される
            connections = [
                ('x_pos', 'y_pos'), ('x_pos', 'y_neg'), ('x_pos', 'z_pos'), ('x_pos', 'z_neg'),
                ('x_neg', 'y_pos'), ('x_neg', 'y_neg'), ('x_neg', 'z_pos'), ('x_neg', 'z_neg'),
                ('y_pos', 'z_pos'), ('y_pos', 'z_neg'),
                ('y_neg', 'z_pos'), ('y_neg', 'z_neg')
            ]
            
            for conn1, conn2 in connections:
                idx1 = o_indices_by_axis.get(conn1)
                idx2 = o_indices_by_axis.get(conn2)
                if idx1 is not None and idx2 is not None:
                    o1 = o_vertices[idx1]
                    o2 = o_vertices[idx2]
                    ax.plot([o1[0], o2[0]], 
                           [o1[1], o2[1]], 
                           [o1[2], o2[2]], 
                           'k-', linewidth=1, alpha=0.3)
    
    # Bサイト間の結合線を描画（隣接するTi原子間）
    # 1ユニットセルの場合、Bサイトは1つだけなので結合線は不要
    # ただし、将来的に複数ユニットセルに対応する場合に備えて実装
    if len(ti_positions) > 1:
        # 隣接するBサイト間の距離を計算して結合線を描画
        # 格子定数a=1.0の場合、隣接するBサイト間の距離は1.0
        for i in range(len(ti_positions)):
            for j in range(i+1, len(ti_positions)):
                ti1 = ti_positions[i]
                ti2 = ti_positions[j]
                dist = np.linalg.norm(ti2 - ti1)
                # 隣接するBサイト（距離が約1.0）のみ結合線を描画
                if abs(dist - 1.0) < 0.1:
                    ax.plot([ti1[0], ti2[0]], 
                           [ti1[1], ti2[1]], 
                           [ti1[2], ti2[2]], 
                           'r--', linewidth=1.5, alpha=0.5)
    
    # Aサイト間の結合線を描画（立方体のワイヤーフレーム）
    # 各ユニットセルの8つの角のBa原子を接続して立方体のエッジを描画
    # 1ユニットセルの場合、8つのBa原子が立方体の角に位置
    # 立方体の12本のエッジを描画（対角線は描画しない）
    if len(ba_positions) >= 8:
        # 立方体のエッジ：2つの角の原子が1つの座標のみが異なる場合のみ接続
        for i in range(len(ba_positions)):
            for j in range(i+1, len(ba_positions)):
                ba1 = ba_positions[i]
                ba2 = ba_positions[j]
                diff = np.abs(ba2 - ba1)
                # 1つの座標のみが異なり、その差が約1.0の場合（エッジ）
                # 対角線（2つ以上の座標が異なる）は除外
                if np.sum(diff > 0.1) == 1 and np.max(diff) < 1.1:
                    ax.plot([ba1[0], ba2[0]], 
                           [ba1[1], ba2[1]], 
                           [ba1[2], ba2[2]], 
                           'gray', linewidth=1.5, alpha=0.4, linestyle='-')
    
    if not simple_mode:
        p_label = "P'" if APPLY_UNIT_CONVERSION else "P"
        r_label = "R'" if APPLY_UNIT_CONVERSION else "R"
        ax.set_title(f'BaTiO3 Unit Cell (Oblique View)\n{p_label}={P:.3f}, {r_label}={R:.3f}, E={E:.3f}')
        ax.legend(loc='upper right', fontsize=8, markerscale=0.6, framealpha=0.9, fancybox=True)
    
    # アスペクト比を1:1:1に設定（立方晶を正しく表示）
    # データ範囲を取得してアスペクト比を設定
    all_positions = np.vstack([ba_positions, ti_positions, o_positions])
    x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
    y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
    z_range = all_positions[:, 2].max() - all_positions[:, 2].min()
    
    # 最大範囲に合わせてアスペクト比を設定
    max_range = max(x_range, y_range, z_range)
    if max_range > 0:
        ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
    else:
        ax.set_box_aspect([1, 1, 1])
    
    # 軸とメッシュを非表示（シンプルモードでも非表示）
    ax.set_axis_off()
    
    # 視点を設定（斜めから見る）
    ax.view_init(elev=15, azim=20)
    
    # パラメータテキストを追加
    if not simple_mode:
        ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes,
                  fontsize=8, verticalalignment='top', horizontalalignment='left',
                  bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
    
    # 背景を白に設定して透明度を正しくラスタライズ
    fig.patch.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # PNG保存時に背景を白に設定し、透明度を正しく保存
    # 3Dプロットの透明度を正しくラスタライズするため、一度描画してから保存
    plt.draw()
    # ディレクトリはメインループで既に作成済み
    plt.savefig(os.path.join(subdir, f"3Dunitcell_oblique_{n:04d}.png"), 
                bbox_inches='tight', dpi=150, facecolor='white', edgecolor='none', 
                transparent=False, pad_inches=0.1)
    plt.close()

def plot_3d_unitcell_topview(ba_positions, ti_positions, o_positions, octahedra_info,
                               P, R, E, n, subdir, param_text="", o_positions_original=None,
                               ti_positions_prev=None, o_positions_prev=None, simple_mode=None):
    """
    【軽量化版】視点2：2×2×1ユニットセルをC軸垂直方向から見た2D投影
    3D処理を廃止し、純粋な2D描画を行うことで高速化・省メモリ化を実現
    """
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
        
    # 3Dではなく通常の2Dプロットを作成（これで劇的に軽くなります）
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 原子のサイズ（2D用に少し調整）
    ba_size = 500
    ti_size = 500
    o_size = 300
    
    # --- 1. 静的な要素（Ba原子）の描画 ---
    # BaはAサイト固定として最背面に描画
    ax.scatter(ba_positions[:, 0], ba_positions[:, 1],
               s=ba_size, c='green', alpha=0.4, label='Ba (A-site)', 
               edgecolors='darkgreen', linewidths=1, zorder=1)

    # Ba原子の結合線を描画（ユニットセルの外周と内部のエッジ）
    # 立方体のワイヤーフレーム：2つの角の原子が1つの座標のみが異なる場合のみ接続
    # 2D投影のため、z座標は無視してXY平面で判定
    # 重複を避けるため、各Ba原子についてチェック
    if len(ba_positions) >= 8:
        drawn_edges = set()  # 描画済みのエッジを記録（重複防止）
        for i in range(len(ba_positions)):
            for j in range(i+1, len(ba_positions)):
                ba1 = ba_positions[i]
                ba2 = ba_positions[j]
                # XY平面での差分を計算（z座標は無視）
                xy_diff = np.abs(ba2[:2] - ba1[:2])
                # 1つの座標のみが異なり、その差が約格子定数（1.0）の場合（エッジ）
                # 対角線（2つ以上の座標が異なる）は除外
                if np.sum(xy_diff > 0.1) == 1 and np.max(xy_diff) < 1.1:
                    # エッジのキーを生成（順序を統一して重複チェック）
                    edge_key = tuple(sorted([(ba1[0], ba1[1]), (ba2[0], ba2[1])]))
                    if edge_key not in drawn_edges:
                        ax.plot([ba1[0], ba2[0]], 
                               [ba1[1], ba2[1]], 
                               'gray', linewidth=2, alpha=0.5, linestyle='-', zorder=0)
                        drawn_edges.add(edge_key)
    
    # O原子の結合線を描画（Ti-O結合と隣接O原子間の結合）
    # Ti-O結合線：各Ti原子からその周りのO原子への結合線
    for ti_idx, o_indices in octahedra_info:
        if ti_idx < len(ti_positions):
            ti_center = ti_positions[ti_idx]
            for o_idx in o_indices:
                if o_idx < len(o_positions):
                    o_pos = o_positions[o_idx]
                    ax.plot([ti_center[0], o_pos[0]], 
                           [ti_center[1], o_pos[1]], 
                           'blue', linewidth=1, alpha=0.3, linestyle='-', zorder=1)
    
    # 隣接するO原子間の結合線（八面体のエッジ）
    # 同じTi原子に属するO原子間で、距離が近いものを結合
    for ti_idx, o_indices in octahedra_info:
        if ti_idx < len(ti_positions) and len(o_indices) >= 6:
            o_vertices = o_positions[o_indices]
            # 各O原子ペア間の距離を計算
            for i in range(len(o_indices)):
                for j in range(i+1, len(o_indices)):
                    o1 = o_vertices[i]
                    o2 = o_vertices[j]
                    # XY平面での距離を計算
                    xy_dist = np.linalg.norm(o2[:2] - o1[:2])
                    # 距離が適切な範囲内（八面体のエッジの長さ）の場合に結合
                    # 格子定数1.0の八面体では、エッジの長さは約0.707（√2/2）から約1.0の範囲
                    if 0.5 < xy_dist < 1.2:
                        ax.plot([o1[0], o2[0]], 
                               [o1[1], o2[1]], 
                               'blue', linewidth=1, alpha=0.3, linestyle='-', zorder=1)



    # --- 2. 八面体の描画（2Dポリゴン化による軽量化） ---
    # 3Dの8面体を描く代わりに、Tiを中心とした赤道面の酸素4つを結ぶ四角形を描画
    patches = []
    for ti_idx, o_indices in octahedra_info:
        if ti_idx < len(ti_positions) and len(o_indices) >= 6:
            ti_center = ti_positions[ti_idx]
            o_vertices = o_positions[o_indices]
            
            # Tiと同じ高さ(z座標)にある酸素原子（赤道面）を抽出
            # z座標の差が小さいものを選ぶ
            z_diff = np.abs(o_vertices[:, 2] - ti_center[2])
            equatorial_indices = np.where(z_diff < 0.4)[0] # 格子定数の半分より十分小さい閾値
            
            if len(equatorial_indices) >= 4:
                eq_oxygens = o_vertices[equatorial_indices]
                # 中心からの角度でソートして、きれいな四角形を作る
                rel_pos = eq_oxygens[:, :2] - ti_center[:2]
                angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
                sort_order = np.argsort(angles)
                sorted_xy = eq_oxygens[sort_order, :2]
                
                # ポリゴンを作成
                poly = Polygon(sorted_xy, closed=True, fill=True, 
                              facecolor='lightblue', alpha=0.4, 
                              edgecolor='blue', linewidth=0.5)
                ax.add_patch(poly)



    # --- 3. 動的な要素（残像） ---
    # 元の位置のO原子を薄く表示（回転が0の位置）
    if o_positions_original is not None:
        ax.scatter(o_positions_original[:, 0], o_positions_original[:, 1],
                   s=o_size, c='lightblue', alpha=0.3, edgecolors='steelblue', 
                   linewidths=0.3, zorder=1)
    
    if ti_positions_prev is not None:
        # Tiの軌跡（グラデーション線）
        if len(ti_positions_prev) == len(ti_positions):
            for i in range(len(ti_positions)):
                # 線を細かく分割してグラデーションを適用
                num_segments = 5
                x_segments = np.linspace(ti_positions_prev[i, 0], ti_positions[i, 0], num_segments + 1)
                y_segments = np.linspace(ti_positions_prev[i, 1], ti_positions[i, 1], num_segments + 1)
                # 現在位置側を濃く、前の位置側を薄くする
                for j in range(num_segments):
                    alpha = 0.6 * (j + 1) / num_segments  # 0から0.6までグラデーション
                    ax.plot([x_segments[j], x_segments[j+1]],
                           [y_segments[j], y_segments[j+1]],
                           'r-', linewidth=8, alpha=alpha, zorder=2)
            
    if o_positions_prev is not None:
        # Oの軌跡（グラデーション線で回転が見えるように）
        if len(o_positions_prev) == len(o_positions):
            for i in range(len(o_positions)):
                # 線を細かく分割してグラデーションを適用
                num_segments = 5
                x_segments = np.linspace(o_positions_prev[i, 0], o_positions[i, 0], num_segments + 1)
                y_segments = np.linspace(o_positions_prev[i, 1], o_positions[i, 1], num_segments + 1)
                # 現在位置側を濃く、前の位置側を薄くする
                for j in range(num_segments):
                    alpha = 0.5 * (j + 1) / num_segments  # 0から0.5までグラデーション
                    ax.plot([x_segments[j], x_segments[j+1]],
                           [y_segments[j], y_segments[j+1]],
                           'b-', linewidth=6, alpha=alpha, zorder=2)



    # --- 4. 動的な要素（現在の原子位置） ---
    # Ti原子（Bサイト）
    ax.scatter(ti_positions[:, 0], ti_positions[:, 1],
               s=ti_size, c='red', alpha=1.0, label='Ti (B-site)', 
               edgecolors='darkred', linewidths=1, zorder=4)
    
    # O原子（Xサイト）
    ax.scatter(o_positions[:, 0], o_positions[:, 1],
               s=o_size, c='blue', alpha=0.8, label='O (X-site)', 
               edgecolors='darkblue', linewidths=0.5, zorder=3)



    # 結合線（Ti-O）
    # ループ処理を排除し、MatplotlibのLineCollectionを使うともっと軽くなるが
    # ここでは単純化のため、主要な結合のみ描画
    # (八面体のポリゴンで代用できているので、線は薄くする)



    # --- 5. レイアウト設定 ---
    if not simple_mode:
        p_label = "P'" if APPLY_UNIT_CONVERSION else "P"
        r_label = "R'" if APPLY_UNIT_CONVERSION else "R"
        ax.set_title(f'BaTiO3 Supercell (2×2×1) Top View [2D Mode]\n{p_label}={P:.3f}, {r_label}={R:.3f}, E={E:.3f}')
        ax.legend(loc='upper right', fontsize=8, markerscale=0.6, framealpha=0.9, fancybox=True)
    
    ax.set_aspect('equal')
    
    # 表示範囲の設定（マージン確保）
    all_x = np.concatenate([ba_positions[:, 0], ti_positions[:, 0], o_positions[:, 0]])
    all_y = np.concatenate([ba_positions[:, 1], ti_positions[:, 1], o_positions[:, 1]])
    
    margin = 0.5
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    ax.axis('off')



    # パラメータテキスト
    if not simple_mode:
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))



    # 保存（ディレクトリはメインループで既に作成済み）
    # 高解像度で保存
    plt.savefig(os.path.join(subdir, f"3Dunitcell_topview_{n:04d}.png"), 
                bbox_inches='tight', dpi=150)
    plt.close()

def create_animation(root_folder, output_filename, pattern, start_index=0, pbar=None, total_duration=None, use_fps=False):
    file_list = []
    # サブディレクトリは作らない方針に変更したため、os.listdirで十分
    if not os.path.isdir(root_folder):
        tqdm.write(f"Error: Directory not found: {root_folder}")
        return
    
    for f in os.listdir(root_folder):
        if pattern in f and f.endswith(".png"):
            file_list.append(f)

    def sort_key_by_number(filename):
        # ファイル名から連番を抽出してソートキーとする
        match = re.search(r'(\d{4})\.png$', filename)
        if match:
            return int(match.group(1))
        return -1 

    file_list.sort(key=sort_key_by_number)

    if not file_list:
        tqdm.write(f"Warning: No images found in {root_folder} with pattern '{pattern}'")
        return

    # 開始インデックス以降のファイルのみを抽出（速度優先モードで2周目のみの場合）
    if start_index > 0:
        filtered_list = []
        for f in file_list:
            match = re.search(r'(\d{4})\.png$', f)
            if match:
                frame_number = int(match.group(1))
                if frame_number >= start_index:
                    filtered_list.append(f)
        file_list = filtered_list

    if not file_list:
        tqdm.write(f"Warning: No images found starting from index {start_index} in {root_folder} with pattern '{pattern}'")
        return

    # imageioに渡すためにフルパスに変換
    full_path_list = [os.path.join(root_folder, f) for f in file_list]

    output_path = os.path.join(root_folder, output_filename)
    
    # GIF作成
    # fpsからduration（秒）に変換して使用（GIFの仕様に合わせる）
    fps = 10  # 両モード共通でfps=10
    # 3Dユニットセルの場合はfpsを半分にする
    if use_fps:
        fps = fps / 2.0
    if not use_fps:
        duration = 1.0 / fps  # 各フレームの表示時間（秒）
    
    # 画像読み込み処理を共通化（リトライ機能付き）
    def load_image_as_rgb(filename, max_retries=3, retry_delay=0.1):
        """画像を読み込んでRGB形式に変換（RGBAの場合は白背景に合成）
        ファイルが完全に書き込まれるまで待つリトライ機能付き"""
        from PIL import Image
        for attempt in range(max_retries):
            if not os.path.exists(filename):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                tqdm.write(f"Warning: File not found: {filename}")
                return None
            
            # ファイルサイズが0でないことを確認
            try:
                file_size = os.path.getsize(filename)
                if file_size == 0:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    tqdm.write(f"Warning: File is empty (0 bytes): {filename}")
                    return None
            except OSError as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                tqdm.write(f"Warning: Cannot access file size: {filename}, error: {e}")
                return None
            
            try:
                img = Image.open(filename)
                # 画像が完全に読み込めるか確認
                img.verify()
                # verify()後は画像を再度開く必要がある
                img = Image.open(filename)
                
                if img.mode == 'RGBA':
                    # 白背景のRGB画像を作成
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])  # アルファチャンネルをマスクとして使用
                    return np.array(rgb_img)
                else:
                    return np.array(img)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                tqdm.write(f"Warning: Error processing {filename} (attempt {attempt + 1}/{max_retries}): {e}")
                return None
        
        return None
    
    try:
        # writerの作成パラメータ
        # use_fps=Trueの場合はfpsを直接使用、Falseの場合はdurationを使用
        if use_fps:
            writer_kwargs = {'mode': 'I', 'fps': fps}
        else:
            writer_kwargs = {'mode': 'I', 'duration': duration}
        
        # 読み込み成功・失敗のカウンター
        success_count = 0
        failed_files = []
        
        with imageio.get_writer(output_path, **writer_kwargs) as writer:
            for idx, filename in enumerate(full_path_list):
                image = load_image_as_rgb(filename)
                if image is not None:
                    writer.append_data(image)
                    success_count += 1
                else:
                    failed_files.append(os.path.basename(filename))
                    tqdm.write(f"Failed to load image {idx + 1}/{len(full_path_list)}: {os.path.basename(filename)}")
        
        # 結果の報告
        total_files = len(full_path_list)
        if failed_files:
            tqdm.write(f"Warning: {len(failed_files)}/{total_files} images failed to load for pattern '{pattern}'")
            tqdm.write(f"Failed files: {', '.join(failed_files[:10])}{'...' if len(failed_files) > 10 else ''}")
        else:
            tqdm.write(f"Successfully loaded all {total_files} images for pattern '{pattern}'")
        
        # GIFファイルが正常に作成されたか確認
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                tqdm.write(f"GIF saved successfully: {output_path} ({file_size} bytes, {success_count}/{total_files} frames)")
            else:
                tqdm.write(f"Error: GIF file created but is empty: {output_path}")
                try:
                    os.remove(output_path)
                except:
                    pass
        else:
            tqdm.write(f"Error: GIF file was not created: {output_path}")
        
        # GIF生成後にPNGファイルを自動削除（全モード共通）
        if AUTO_DELETE_PNG_AFTER_GIF:
            deleted_count = 0
            # バッチ削除で高速化
            for filename in full_path_list:
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                        deleted_count += 1
                except OSError as e:
                    tqdm.write(f"Error deleting {filename}: {e}")
            if deleted_count > 0:
                tqdm.write(f"Deleted {deleted_count} PNG files after GIF creation")
    except Exception as e:
        tqdm.write(f"Error creating GIF: {e}")
        import traceback
        tqdm.write(traceback.format_exc())

def create_summary_gif(root_folder, output_filename, start_index=0, pbar=None):
    """全プロットタイプを一覧表示するGIFを作成（画像読み込み版 - レガシー）"""
    from matplotlib.gridspec import GridSpec
    
    # 各プロットタイプのファイルリストを取得
    patterns = []
    if GENERATE_2D_CONTOUR:
        patterns.append(("2Dcontour_", "2D Contour"))
    if GENERATE_3D_SURFACE:
        patterns.append(("3Dsurf_", "3D Surface"))
    if GENERATE_CROSS_SECTION_P:
        patterns.append(("cross_section_P_", "Cross-Section P"))
    if GENERATE_CROSS_SECTION_R:
        patterns.append(("cross_section_R_", "Cross-Section R"))
    if GENERATE_HYSTERESIS_ANIMATION:
        patterns.append(("hysteresis_P_E_", "P-E Hysteresis"))
        patterns.append(("hysteresis_R_E_", "R-E Hysteresis"))
    
    if not patterns:
        print("No plot types enabled for summary GIF")
        return
    
    # ファイルリストを整理
    all_frames_by_type = {}
    max_frames = 0
    
    for pattern, label in patterns:
        file_list = []
        for f in os.listdir(root_folder):
            if pattern in f and f.endswith(".png"):
                file_list.append(f)
        
        # ファイル名から番号を抽出してソート
        def sort_key_by_number(filename):
            match = re.search(r'(\d{4})\.png$', filename)
            if match:
                return int(match.group(1))
            return -1
        
        file_list.sort(key=sort_key_by_number)
        
        # 開始インデックス以降のファイルのみを抽出
        if start_index > 0:
            filtered_list = []
            for f in file_list:
                match = re.search(r'(\d{4})\.png$', f)
                if match:
                    frame_number = int(match.group(1))
                    if frame_number >= start_index:
                        filtered_list.append(f)
            file_list = filtered_list
        
        if file_list:
            all_frames_by_type[pattern] = {
                'files': [os.path.join(root_folder, f) for f in file_list],
                'label': label
            }
            max_frames = max(max_frames, len(file_list))
    
    if not all_frames_by_type:
        print("No images found for summary GIF")
        return
    
    # レイアウトを計算（2列×3行まで）
    num_plots = len(all_frames_by_type)
    if num_plots <= 2:
        nrows, ncols = 1, num_plots
    elif num_plots <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 3, 2
    
    output_path = os.path.join(root_folder, output_filename)
    
    # GIF作成
    # fpsからduration（秒）に変換して使用（GIFの仕様に合わせる）
    fps = 10  # 両モード共通でfps=10
    duration = 1.0 / fps  # 各フレームの表示時間（秒）
    
    try:
        writer = imageio.get_writer(output_path, mode='I', duration=duration)
        
        for frame_idx in range(max_frames):
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(nrows, ncols, figure=fig, wspace=0.15, hspace=0.15)
            
            plot_idx = 0
            for pattern, info in all_frames_by_type.items():
                files = info['files']
                label = info['label']
                
                if frame_idx < len(files):
                    row = plot_idx // ncols
                    col = plot_idx % ncols
                    ax = fig.add_subplot(gs[row, col])
                    
                    # 画像を読み込んで表示
                    img = plt.imread(files[frame_idx])
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
                    
                    plot_idx += 1
            
            # 残りのスペースを空白で埋める（もしあれば）
            while plot_idx < nrows * ncols:
                row = plot_idx // ncols
                col = plot_idx % ncols
                ax = fig.add_subplot(gs[row, col])
                ax.axis('off')
                plot_idx += 1
            
            # 全体のタイトル
            fig.suptitle(f"Summary of All Plots - Frame {frame_idx + 1}/{max_frames}", 
                        fontsize=14, fontweight='bold', y=0.98)
            
            # メモリ効率のため一時ファイルに保存してから読み込む
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            image = imageio.imread(buf)
            writer.append_data(image)
            buf.close()
            plt.close(fig)
        
        writer.close()
        
        # GIFファイルが正常に作成されたか確認
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                tqdm.write(f"Summary GIF saved successfully: {output_path} ({file_size} bytes, {max_frames} frames)")
            else:
                tqdm.write(f"Error: Summary GIF file created but is empty: {output_path}")
                try:
                    os.remove(output_path)
                except:
                    pass
        else:
            tqdm.write(f"Error: Summary GIF file was not created: {output_path}")
        
    except Exception as e:
        tqdm.write(f"Error creating summary GIF: {e}")
        import traceback
        tqdm.write(traceback.format_exc())

# ------------------------------------------------------------------
if __name__ == "__main__":
    # 実行開始時刻を記録
    start_time = time.time()
    
    # Outputsフォルダが存在しない場合は作成
    if not os.path.isdir(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
    
    # temp_E_sweepフォルダのみをリセット
    if os.path.isdir(GRAPH_SAVE_DIR):
        shutil.rmtree(GRAPH_SAVE_DIR)
    os.makedirs(GRAPH_SAVE_DIR)

    P_values = np.arange(P_grid[0], P_grid[1] + grid_mesh, grid_mesh)
    R_values = np.arange(R_grid[0], R_grid[1] + grid_mesh, grid_mesh)
    P_mesh, R_mesh = np.meshgrid(P_values, R_values, indexing='ij')

    print(f"Running in {'SPEED MODE' if SPEED_MODE else 'QUALITY MODE'}")
    print(f"GENERATE_3D_UNITCELL = {GENERATE_3D_UNITCELL}")
    
    # 各工程ごとのプログレスバーを使用するため、重み付け計算は不要
    n_1st = len(E_range_1st)  # 1周目の電場数
    n_2nd = len(E_range_2nd)  # 2周目の電場数
    total_e_steps = len(E_range)
    num_plot_frames = n_2nd if SKIP_FIRST_CYCLE_PLOT else total_e_steps
    
    # GIFパターンのリストを作成（GIF作成時のプログレスバー用）
    gif_patterns = []
    if GENERATE_SUMMARY_GIF:
        gif_patterns.append("summary_")
    if GENERATE_3D_SURFACE:
        gif_patterns.append("3Dsurf_")
    if GENERATE_2D_CONTOUR:
        gif_patterns.append("2Dcontour_")
    if GENERATE_CROSS_SECTION_P:
        gif_patterns.append("cross_section_P_")
    if GENERATE_CROSS_SECTION_R:
        gif_patterns.append("cross_section_R_")
    if GENERATE_HYSTERESIS_ANIMATION:
        gif_patterns.append("hysteresis_P_E_")
        gif_patterns.append("hysteresis_R_E_")
    if GENERATE_3D_UNITCELL:
        gif_patterns.append("3Dunitcell_oblique_")
        gif_patterns.append("3Dunitcell_topview_")
    
    # ターミナル出力のバッファリングを無効化（プログレスバーが確実に表示されるように）
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except:
            pass
    
    tqdm.write("Calculating z-axis range for all plots...")
    
    # モードに応じて計算方法を切り替え
    if SPEED_MODE:
        # ベクトル化された計算で高速化（プログレスバー不要）
        E_range_expanded = E_range[:, np.newaxis, np.newaxis]  # (n_E, 1, 1)
        P_mesh_expanded = P_mesh[np.newaxis, :, :]  # (1, n_P, n_R)
        R_mesh_expanded = R_mesh[np.newaxis, :, :]  # (1, n_P, n_R)
        
        # 全電場での自由エネルギーを一度に計算
        T = fixed_params['T_FIXED']
        alpha_1_t = fixed_params['alpha_1'] * (T - fixed_params['t_{cP}'])
        beta_1_eff = fixed_params['beta_1'] * (T - fixed_params['t_{cR}'])
        
        # 単位変換係数の計算
        if APPLY_UNIT_CONVERSION:
            # Pの単位変換: [uC/cm^2] → [C/m^2] のため、各P項の係数を100^(P次数)倍
            # Rの単位変換: 角度定義R（φ）→ 変位定義r = l*φ/2 のため、各R項の係数を(2/l)^(R次数)倍
            # R = r/(l/2) = r/0.2 = 5*r より、R^2 → 25*r^2, R^4 → 625*r^4, R^6 → 15625*r^6
            P_conv_1 = 100**1  # P^1項（電場項）: 100
            P_conv_2 = 100**2  # P^2項: 10000
            P_conv_4 = 100**4  # P^4項: 100000000
            P_conv_6 = 100**6  # P^6項: 1000000000000
            P_conv_8 = 100**8  # P^8項: 10000000000000000
            R_conv_2 = (2.0/LATTICE_CONSTANT_L)**2  # R^2項: 25
            R_conv_4 = (2.0/LATTICE_CONSTANT_L)**4  # R^4項: 625
            R_conv_6 = (2.0/LATTICE_CONSTANT_L)**6  # R^6項: 15625
        else:
            # 単位変換を適用しない場合、係数は1
            P_conv_1 = P_conv_2 = P_conv_4 = P_conv_6 = P_conv_8 = 1.0
            R_conv_2 = R_conv_4 = R_conv_6 = 1.0
        
        all_energies = (
            0.5 * alpha_1_t * P_mesh_expanded**2 * P_conv_2  # P^2項
            + 0.25 * fixed_params['alpha_2'] * P_mesh_expanded**4 * P_conv_4  # P^4項
            + (1.0/6.0) * fixed_params['alpha_3'] * P_mesh_expanded**6 * P_conv_6  # P^6項
            + 0.5 * beta_1_eff * R_mesh_expanded**2 * R_conv_2  # R^2項
            + 0.25 * fixed_params['beta_2'] * R_mesh_expanded**4 * R_conv_4  # R^4項
            + (1.0/6.0) * fixed_params['beta_3'] * R_mesh_expanded**6 * R_conv_6  # R^6項
            + fixed_params['gamma_22'] * P_mesh_expanded**2 * R_mesh_expanded**2 * P_conv_2 * R_conv_2  # P^2*R^2項
            + fixed_params['gamma_42'] * P_mesh_expanded**4 * R_mesh_expanded**2 * P_conv_4 * R_conv_2  # P^4*R^2項
            + fixed_params['gamma_82'] * P_mesh_expanded**8 * R_mesh_expanded**2 * P_conv_8 * R_conv_2  # P^8*R^2項
            + fixed_params['gamma_64'] * P_mesh_expanded**6 * R_mesh_expanded**4 * P_conv_6 * R_conv_4  # P^6*R^4項
            + fixed_params['gamma_84'] * P_mesh_expanded**8 * R_mesh_expanded**4 * P_conv_8 * R_conv_4  # P^8*R^4項
            + fixed_params['gamma_86'] * P_mesh_expanded**8 * R_mesh_expanded**6 * P_conv_8 * R_conv_6  # P^8*R^6項
            - P_mesh_expanded * E_range_expanded * P_conv_1  # P^1項（電場項）
        )
        
        # エネルギー値のカットオフ
        all_energies_cut = np.where(all_energies > ENERGY_CUTOFF, ENERGY_CUTOFF, all_energies)
        
        z_min = all_energies_cut.min()
        z_max = all_energies_cut.max()
        
        # メモリ効率化：不要な変数を削除
        del E_range_expanded, P_mesh_expanded, R_mesh_expanded
        sys.stdout.flush()  # 出力を確実にフラッシュ
    else:
        # 品質優先モード：計算
        all_energies = []
        for e in E_range:
            all_energies.append(calc_free_energy(P_mesh, R_mesh, e, fixed_params))
        
        # エネルギー上限をカットしてレンジを決定
        all_energies_cut = []
        for e in all_energies:
            all_energies_cut.append(np.where(e > ENERGY_CUTOFF, ENERGY_CUTOFF, e))
        z_min = min(e.min() for e in all_energies_cut)
        z_max = max(e.max() for e in all_energies_cut)
    
    tqdm.write(f"Global z-range set to: ({z_min:.2f}, {z_max:.2f}) (with cutoff at {ENERGY_CUTOFF})")

    E_array = []
    P_stable_array = []
    R_stable_array = []

    current_p = init_P_value
    current_r = init_R_value
    prev_p = None
    prev_r = None
    
    # 3Dモデル描画用：前のフレームの原子位置を保存（品質優先モードの残像表示用）
    ti_pos_1_prev = None
    o_pos_1_prev = None
    ti_pos_2_prev = None
    o_pos_2_prev = None
    
    # 3Dモデル描画用：PとRの最大値を追跡（2周目のデータのみ）
    P_max = 0.0
    R_max = 0.0
    
    # 変数名を下付き文字で表示するための変換
    def format_param_name(param_name):
        # 下付き文字の変換辞書
        subscript_map = {
            'T_FIXED': 'T_{fixed}',
            't_{cP}': 't_{cP}',
            't_{cR}': 't_{cR}',
            'alpha_1': 'α_1',
            'alpha_2': 'α_2',
            'alpha_3': 'α_3',
            'beta_1': 'β_1',
            'beta_2': 'β_2',
            'beta_3': 'β_3',
            'gamma_22': 'γ_{22}',
            'gamma_42': 'γ_{42}',
            'gamma_82': 'γ_{82}',
            'gamma_64': 'γ_{64}',
            'gamma_84': 'γ_{84}',
            'gamma_86': 'γ_{86}'
        }
        return subscript_map.get(param_name, param_name)
    
    # format_value_with_colorを使用（重複関数を削除）
    # γ項は変数値が0の時は表示しない、α・β項は0でも表示する
    param_text_for_plots = "\n".join([
        f"${format_param_name(k)}$ = {format_value_with_color(v)}" 
        for k, v in fixed_params.items() 
        if not (k.startswith('gamma_') and v == 0)
    ])
    
    # 3Dモデル描画用：2周目の全データを先に収集してPとRの最大値を計算
    tqdm.write("Calculating P and R maximum values for 3D model scaling...")
    P_max_temp = 0.0
    R_max_temp = 0.0
    temp_p = init_P_value
    temp_r = init_R_value
    # 2周目の範囲のみを処理
    E_range_2nd_for_max = E_range[n_1st:]
    for n_temp, e_temp in enumerate(E_range_2nd_for_max, start=n_1st):
        free_energy_temp = all_energies_cut[n_temp]  # SPEED_MODEとQUALITY_MODEで同じ処理
        # 簡易的な安定点探索
        stable_p_temp, stable_r_temp = find_stable_point(P_mesh, R_mesh, free_energy_temp, temp_p, temp_r)
        temp_p = stable_p_temp
        temp_r = stable_r_temp
        P_max_temp = max(P_max_temp, abs(stable_p_temp))
        R_max_temp = max(R_max_temp, abs(stable_r_temp))
    # 最大値が0の場合はデフォルト値を設定
    P_max = P_max_temp if P_max_temp > 0 else 1.0
    R_max = R_max_temp if R_max_temp > 0 else 1.0
    tqdm.write(f"P_max = {P_max:.3f}, R_max = {R_max:.3f}")
    
    # メインループ：全範囲で計算を行い、描画のみを制御する
    tqdm.write("Running main simulation loop...")
    main_pbar = tqdm(total=total_e_steps, desc="Main simulation", 
                     mininterval=0.1, maxinterval=1.0, leave=True)
    
    for n, e in enumerate(E_range):
        free_energy = all_energies_cut[n]  # カットオフ後のエネルギーを利用（両モード共通）
        
        stable_p, stable_r = find_stable_point(P_mesh, R_mesh, free_energy, current_p, current_r)
        
        # ---------------------------------------------------------
        # 【修正1】数値発散（NaN/Inf）のガード処理
        # 計算結果がおかしい場合は、前の値か0.0を採用してエラーを防ぐ
        # ---------------------------------------------------------
        if not np.isfinite(stable_p) or not np.isfinite(stable_r):
            tqdm.write(f"Warning: NaN/Inf detected at frame {n} (E={e:.3f}). Using previous values.")
            if prev_p is not None and np.isfinite(prev_p) and np.isfinite(prev_r):
                stable_p = prev_p
                stable_r = prev_r
            else:
                stable_p = 0.0
                stable_r = 0.0

        E_array.append(e)
        P_stable_array.append(stable_p)
        R_stable_array.append(stable_r)

        # 両モード共通：1周目は計算のみ、2周目からグラフ描画
        should_plot = not SKIP_FIRST_CYCLE_PLOT or n >= n_1st
        
        if should_plot:
            # 2周目のインデックスに調整（アニメーション用の連番）
            graph_n = (n - n_1st) if SKIP_FIRST_CYCLE_PLOT else n
            
            # 各フレームのグラフをプロット（前の安定点座標も渡す）
            if GENERATE_2D_CONTOUR:
                plot_2d_contour(P_mesh, R_mesh, free_energy, stable_p, stable_r, e, graph_n, GRAPH_SAVE_DIR, fixed_params, z_lim=(z_min, z_max), param_text=param_text_for_plots, prev_stable_p=prev_p if SHOW_TRAJECTORY else None, prev_stable_r=prev_r if SHOW_TRAJECTORY else None, simple_mode=SIMPLE_PLOT_MODE)
            if GENERATE_3D_SURFACE:
                plot_3d_surface(P_mesh, R_mesh, free_energy, stable_p, stable_r, e, graph_n, GRAPH_SAVE_DIR, fixed_params, z_lim=(z_min, z_max), param_text=param_text_for_plots, simple_mode=SIMPLE_PLOT_MODE)
            # 断面プロットを生成
            if GENERATE_CROSS_SECTION_P:
                plot_cross_section_P(P_mesh, R_mesh, free_energy, stable_p, stable_r, e, graph_n, GRAPH_SAVE_DIR, fixed_params, z_lim=(z_min, z_max), param_text=param_text_for_plots, simple_mode=SIMPLE_PLOT_MODE)
            if GENERATE_CROSS_SECTION_R:
                plot_cross_section_R(P_mesh, R_mesh, free_energy, stable_p, stable_r, e, graph_n, GRAPH_SAVE_DIR, fixed_params, z_lim=(z_min, z_max), param_text=param_text_for_plots, simple_mode=SIMPLE_PLOT_MODE)
            
            # 3Dユニットセルモデルを描画
            if GENERATE_3D_UNITCELL:
                if graph_n == 0:
                    tqdm.write(f"Generating 3D unit cell at frame {graph_n} (E={e:.3f}, P={stable_p:.3f}, R={stable_r:.3f})")
                
                # ---------------------------------------------------------
                # 【修正2】ObliqueとTopViewのTryブロックを分離
                # ---------------------------------------------------------
                
                # --- 視点1：Oblique View ---
                try:
                    ba_pos_1, ti_pos_1, o_pos_1, octa_info_1 = get_perovskite_structure(nx=1, ny=1, nz=1, a=1.0)
                    ti_pos_1_original = ti_pos_1.copy()
                    o_pos_1_original = o_pos_1.copy()
                    ti_pos_1_new, o_pos_1_new = apply_polarization_and_rotation(
                        ba_pos_1, ti_pos_1, o_pos_1, octa_info_1, 
                        stable_p, stable_r, P_max, R_max, a=1.0
                    )
                    plot_3d_unitcell_oblique(
                        ba_pos_1, ti_pos_1_new, o_pos_1_new, octa_info_1,
                        stable_p, stable_r, e, graph_n, GRAPH_SAVE_DIR, param_text=param_text_for_plots,
                        ti_positions_original=ti_pos_1_original, o_positions_original=o_pos_1_original,
                        ti_positions_prev=ti_pos_1_prev,
                        o_positions_prev=o_pos_1_prev,
                        simple_mode=SIMPLE_PLOT_MODE
                    )
                    # 位置更新
                    ti_pos_1_prev = ti_pos_1_new.copy()
                    o_pos_1_prev = o_pos_1_new.copy()
                        
                except Exception as ex:
                    tqdm.write(f"Error generating Oblique unit cell at frame {graph_n}: {ex}")
                    # エラー時も変数が更新されないと次でエラーになるため、前の値を維持またはコピー
                    pass
                
                # --- 視点2：Top View (ここが52枚で止まっていた箇所) ---
                try:
                    ba_pos_2, ti_pos_2, o_pos_2, octa_info_2 = get_perovskite_structure(nx=2, ny=2, nz=1, a=1.0)
                    o_pos_2_original = o_pos_2.copy()
                    ti_pos_2_new, o_pos_2_new = apply_polarization_and_rotation(
                        ba_pos_2, ti_pos_2, o_pos_2, octa_info_2,
                        stable_p, stable_r, P_max, R_max, a=1.0
                    )
                    plot_3d_unitcell_topview(
                        ba_pos_2, ti_pos_2_new, o_pos_2_new, octa_info_2,
                        stable_p, stable_r, e, graph_n, GRAPH_SAVE_DIR, param_text=param_text_for_plots,
                        o_positions_original=o_pos_2_original,
                        ti_positions_prev=ti_pos_2_prev,
                        o_positions_prev=o_pos_2_prev,
                        simple_mode=SIMPLE_PLOT_MODE
                    )
                    # 位置更新
                    ti_pos_2_prev = ti_pos_2_new.copy()
                    o_pos_2_prev = o_pos_2_new.copy()
                except Exception as ex:
                    # エラー内容を詳細に表示
                    tqdm.write(f"Error generating Topview unit cell at frame {graph_n}: {ex}")
                    import traceback
                    traceback.print_exc()
            
            # 各フレームの処理が完了した後に進捗を更新
            main_pbar.update(1)
        else:
            # 1周目をスキップする場合でも、計算処理の進捗を更新
            main_pbar.update(1)
        
        # 次のループのために前の安定点を保存
        prev_p = stable_p
        prev_r = stable_r
        current_p = stable_p
        current_r = stable_r
        
        # ---------------------------------------------------------
        # 【重要】メモリリーク対策：ループ1回ごとに全ての図を破棄してメモリを掃除する
        # ---------------------------------------------------------
        plt.close('all')  # 開いている全てのFigureを強制的に閉じる
        gc.collect()      # ガベージコレクション（メモリ掃除）を強制実行
        # ---------------------------------------------------------
    
    # メインシミュレーションのプログレスバーを閉じる
    main_pbar.close()
    
    # P-E グラフ
    fig, ax = plt.subplots()
    
    # 両モード共通：2回目の電場印加のみ
    if not SIMPLE_PLOT_MODE:
        ax.plot(E_array[n_1st:], P_stable_array[n_1st:], 'o-', markersize=3, label="2nd: P(E)", color='blue')
        ax.set_title("P vs Electric Field (2nd cycle)")
        ax.legend()
        ax.text(1.05, 0.95, param_text_for_plots, transform=ax.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
    else:
        ax.plot(E_array[n_1st:], P_stable_array[n_1st:], 'o-', markersize=3, color='blue')
    
    ax.set_xlabel("E")
    ax.set_ylabel("P'" if APPLY_UNIT_CONVERSION else "P")
    ax.set_ylim(P_E_graph_y)
    ax.grid()  # シンプルモードでもグリッドを表示
    plt.savefig(os.path.join(GRAPH_SAVE_DIR, "P_vs_E.png"), bbox_inches='tight')
    plt.close()

    # R-E グラフ
    fig, ax = plt.subplots()
    
    # 両モード共通：2回目の電場印加のみ
    if not SIMPLE_PLOT_MODE:
        ax.plot(E_array[n_1st:], R_stable_array[n_1st:], 'o-', markersize=3, label="2nd: R(E)", color='blue')
        ax.set_title("R vs Electric Field (2nd cycle)")
        ax.legend()
        ax.text(1.05, 0.95, param_text_for_plots, transform=ax.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
    else:
        ax.plot(E_array[n_1st:], R_stable_array[n_1st:], 'o-', markersize=3, color='blue')
    
    ax.set_xlabel("E")
    ax.set_ylabel("R'" if APPLY_UNIT_CONVERSION else "R")
    ax.set_ylim(R_E_graph_y)
    ax.grid()  # シンプルモードでもグリッドを表示
    plt.savefig(os.path.join(GRAPH_SAVE_DIR, "R_vs_E.png"), bbox_inches='tight')
    plt.close()

    if GENERATE_HYSTERESIS_ANIMATION:
        # 2周目のデータのみを使用
        E_array_2nd = E_array[n_1st:]
        P_stable_array_2nd = P_stable_array[n_1st:]
        R_stable_array_2nd = R_stable_array[n_1st:]
        
        # 各フレームを生成（2周目の各電場値ごと）
        for frame_n, (e_current, p_current, r_current) in enumerate(zip(E_array_2nd, P_stable_array_2nd, R_stable_array_2nd)):
            # 完全なヒステリシスループを表示するため、全データを使用（2周目のデータのみ）
            # current_idxは現在のフレーム番号（frame_n）を使用
            plot_hysteresis_P_E_frame(E_array_2nd, P_stable_array_2nd, e_current, p_current, 
                                      frame_n, frame_n, GRAPH_SAVE_DIR, fixed_params, param_text=param_text_for_plots, simple_mode=SIMPLE_PLOT_MODE)
            plot_hysteresis_R_E_frame(E_array_2nd, R_stable_array_2nd, e_current, r_current, 
                                      frame_n, frame_n, GRAPH_SAVE_DIR, fixed_params, param_text=param_text_for_plots, simple_mode=SIMPLE_PLOT_MODE)

    # 両モード共通：2周目のデータのみでGIF作成
    # graph_nは既に2周目のインデックスに調整されているので0から開始
    start_idx = 0
    gif_suffix = "2nd_cycle" if SPEED_MODE else "2nd"
    
    # 全プロットを一覧表示するGIFを最初に作成（AUTO_DELETE_PNG_AFTER_GIFがTrueの場合、個別GIF作成時にPNGが削除されるため）
    if GENERATE_SUMMARY_GIF:
        tqdm.write("Generating summary GIF with all plot types...")
        create_summary_gif(GRAPH_SAVE_DIR, f"summary_{gif_suffix}.gif", start_idx, pbar=None)
    
    if GENERATE_3D_SURFACE:
        tqdm.write("Generating 3D surface GIF...")
        create_animation(GRAPH_SAVE_DIR, f"3Dsurf_{gif_suffix}.gif", "3Dsurf_", start_idx, pbar=None)
    if GENERATE_2D_CONTOUR:
        tqdm.write("Generating 2D contour GIF...")
        create_animation(GRAPH_SAVE_DIR, f"2Dcontour_{gif_suffix}.gif", "2Dcontour_", start_idx, pbar=None)
    if GENERATE_CROSS_SECTION_P:
        tqdm.write("Generating cross-section P GIF...")
        create_animation(GRAPH_SAVE_DIR, f"cross_section_P_{gif_suffix}.gif", "cross_section_P_", start_idx, pbar=None)
    if GENERATE_CROSS_SECTION_R:
        tqdm.write("Generating cross-section R GIF...")
        create_animation(GRAPH_SAVE_DIR, f"cross_section_R_{gif_suffix}.gif", "cross_section_R_", start_idx, pbar=None)
    
    # ヒステリシスループのアニメーションGIFを作成
    if GENERATE_HYSTERESIS_ANIMATION:
        tqdm.write("Generating hysteresis P-E GIF...")
        create_animation(GRAPH_SAVE_DIR, f"hysteresis_P_E_{gif_suffix}.gif", "hysteresis_P_E_", start_idx, pbar=None)
        tqdm.write("Generating hysteresis R-E GIF...")
        create_animation(GRAPH_SAVE_DIR, f"hysteresis_R_E_{gif_suffix}.gif", "hysteresis_R_E_", start_idx, pbar=None)
    
    # 3DユニットセルモデルのアニメーションGIFを作成
    if GENERATE_3D_UNITCELL:
        tqdm.write("Generating 3D unit cell model animations...")
        try:
            tqdm.write("Generating 3D unit cell oblique GIF...")
            create_animation(GRAPH_SAVE_DIR, f"3Dunitcell_oblique_{gif_suffix}.gif", "3Dunitcell_oblique_", start_idx, pbar=None, use_fps=True)
            tqdm.write("Generating 3D unit cell topview GIF...")
            create_animation(GRAPH_SAVE_DIR, f"3Dunitcell_topview_{gif_suffix}.gif", "3Dunitcell_topview_", start_idx, pbar=None, use_fps=True)
        except Exception as ex:
            tqdm.write(f"Error creating 3D unit cell animations: {ex}")
            import traceback
            traceback.print_exc()
    
    # 実行時間を計算して表示
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    print(f"Simulation completed. Check the '{OUTPUTS_DIR}' folder for outputs.")
    if hours > 0:
        print(f"総実行時間: {hours}時間 {minutes}分 {seconds:.2f}秒")
    elif minutes > 0:
        print(f"総実行時間: {minutes}分 {seconds:.2f}秒")
    else:
        print(f"総実行時間: {seconds:.2f}秒")