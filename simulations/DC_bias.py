#!/usr/bin/env python3
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
# True: 速度優先モード（高速化設定、2周目からグラフ描画）
# False: 品質優先モード（高精度設定）
SPEED_MODE = False

# True: シンプル描画モード（タイトル、ボックス、凡例を非表示、プロットと軸目盛り・軸タイトルのみ表示）
SIMPLE_PLOT_MODE = False

# ------------------------------------------------------------------
# 2. 物理パラメータ（自由エネルギー計算の係数）
# ------------------------------------------------------------------
fixed_params = { 
    'T_FIXED': -50,      # 固定する温度
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
    'gamma_64': -0.000000,      # γ_64*P^6*R^4項の係数
    'gamma_84': 0.0000000005,   # γ_84*P^8*R^4項の係数
    'gamma_86': 0.00000000000,  # γ_86*P^8*R^6項の係数
}

# ------------------------------------------------------------------
# 3. DCバイアス電場設定
# ------------------------------------------------------------------
E_range_abs = 400.0  # 電場範囲の絶対値（最大値は正、最小値は負になる）
E_divisions = 25 if SPEED_MODE else 60  # 電場範囲を何分割するか（スピードモード: 25, 品質モード: 60）
E_max = E_range_abs  # 電場の最大値
E_min = -E_range_abs  # 電場の最小値（負の値）
E_step = 10.0  # DCバイアス測定用の電場ステップ幅

# ------------------------------------------------------------------
# 4. 温度設定（固定温度）
# ------------------------------------------------------------------
# 温度はfixed_params['T_FIXED']で固定されます

# ------------------------------------------------------------------
# 5. 格子・描画範囲設定
# ------------------------------------------------------------------
P_range_abs = 30  # P軸の範囲の絶対値
P_range = (-P_range_abs, P_range_abs)  # P軸の範囲（格子・描画共通）
R_range = (0, 7)  # R軸の範囲（格子・描画共通）
grid_mesh = 0.03  # 格子のメッシュサイズ

# ------------------------------------------------------------------
# 6. エネルギー・カットオフ設定
# ------------------------------------------------------------------
ENERGY_CUTOFF = 10000.0  # エネルギーカットオフ値

# ------------------------------------------------------------------
# 7. 初期値設定
# ------------------------------------------------------------------
init_P_value = 0.0  # Pの初期値
init_R_value = 0.0  # Rの初期値

# ------------------------------------------------------------------
# 8. プロット生成制御フラグ
# ------------------------------------------------------------------
GENERATE_2D_CONTOUR = True      # 2Dコンター図を生成するか
GENERATE_3D_SURFACE = True      # 3Dサーフェス図を生成するか
GENERATE_CROSS_SECTION_P = True # P軸に対する垂直な断面プロットを生成するか
GENERATE_CROSS_SECTION_R = True  # R軸に対する垂直な断面プロットを生成するか
GENERATE_HYSTERESIS_ANIMATION = not SPEED_MODE  # ヒステリシスループのGIFアニメーションを生成する（描画モード: True, スピードモード: False）か
GENERATE_SUMMARY_GIF = False     # 全プロットを一覧表示するGIFを生成するか
GENERATE_3D_UNITCELL = True      # 3DユニットセルモデルのGIFアニメーションを生成するか
SHOW_R_COEFFICIENTS_BOX = True  # R^2, R^4, R^6, Sum, Dのボックスを表示するか
is_show_contour_value = False    # コンター値ラベルを表示するか

# ------------------------------------------------------------------
# 9. 動作制御フラグ（通常は変更不要）
# ------------------------------------------------------------------
SKIP_FIRST_CYCLE_PLOT = True    # 1周目をスキップしてプロットしない
SHOW_TRAJECTORY = not SPEED_MODE  # 軌跡を表示するか（品質優先モードのみ）
AUTO_DELETE_PNG_AFTER_GIF = True  # GIF生成後にPNGを自動削除するか（全モード共通）

# ------------------------------------------------------------------
# 10. 保存先ディレクトリ設定
# ------------------------------------------------------------------
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "Outputs_DC_bias")
# DCバイアス誘電率測定用のディレクトリ
# シンプル描画モードの場合は、dc_bias_speed/dc_bias_qualityの中にsimple_modeディレクトリを作成
# シンプル描画モードがTrueの時はシンプル描画モードの出力のみ、Falseの時は通常モードのみの出力
if SPEED_MODE:
    base_dir = os.path.join(OUTPUTS_DIR, "dc_bias_speed")
else:
    base_dir = os.path.join(OUTPUTS_DIR, "dc_bias_quality")

if SIMPLE_PLOT_MODE:
    GRAPH_SAVE_DIR = os.path.join(base_dir, "simple_mode")
else:
    GRAPH_SAVE_DIR = base_dir

# ==================================================================
# 【自動計算される設定値（変更不要）】
# ==================================================================

# 後方互換性のための設定（P_range, R_rangeから自動設定）
P_grid = P_range
R_grid = R_range
contour_graph_x = P_range
contour_graph_y = R_range
P_E_graph_y = P_range  # P-EプロットのP軸範囲
P_T_graph_y = (0, 30)  # P-TプロットのP軸範囲を0~30に設定
R_T_graph_y = (0, 4)  # R-TプロットのR軸範囲を0~4に設定

# 温度は固定値（fixed_params['T_FIXED']）を使用

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
    
    # matplotlibのmathtextでは\mathbfがサポートされていないため、\bfを使用
    # $は付けずに返す（呼び出し側で適切に結合する）
    if value < 0:
        return f"\\bf{{{formatted}}}"  # 太字でマイナス値を強調（mathtext内で\bfを使用）
    else:
        return formatted  # 通常の値

# ------------------------------------------------------------------
def create_r_coefficients_annotation(R2_val, R4_val, R6_val, total_val, discriminant_D):
    """R^2, R^4, R^6, Sum, Dのアノテーションテキストを生成する共通関数"""
    return (
        f"$R^2$=${format_value_with_color(R2_val)}$\n"
        f"$R^4$=${format_value_with_color(R4_val)}$\n"
        f"$R^6$=${format_value_with_color(R6_val)}$\n"
        f"Sum=${format_value_with_color(total_val)}$\n"
        f"$D$=${format_value_with_color(discriminant_D)}$"
    )

# ------------------------------------------------------------------
def calc_r_coefficients(P_at_stable, T, params):
    """R^2, R^4, R^6の係数値を計算する関数（重複計算を避けるため、温度依存性版）"""
    beta_1_eff = params['beta_1'] * (T - params['t_{cR}'])
    P2 = P_at_stable ** 2
    P4 = P2 ** 2
    P6 = P4 * P2
    P8 = P4 ** 2
    
    R2_val = (
        0.5 * beta_1_eff
        + params['gamma_22'] * P2
        + params['gamma_42'] * P4
    )
    R4_val = (
        0.25 * params['beta_2']
        + params['gamma_64'] * P6
        + params['gamma_84'] * P8
    )
    R6_val = (1.0/6.0) * params['beta_3'] + params['gamma_86'] * P8
    total_val = R2_val + R4_val + R6_val
    
    # 判別式D = c²(b²-4ac) を計算（R6_val=a, R4_val=b, R2_val=c）
    a = R6_val
    b = R4_val
    c = R2_val
    discriminant_D = c**2 * (b**2 - 4*a*c)
    
    return R2_val, R4_val, R6_val, total_val, discriminant_D

# ------------------------------------------------------------------
def calc_free_energy(P, R, T, E, params):
    """自由エネルギーを計算（温度と電場を変数として受け取る）"""
    alpha_1_t = params['alpha_1'] * (T - params['t_{cP}'])
    beta_1_eff = params['beta_1'] * (T - params['t_{cR}'])
    return (
        0.5 * alpha_1_t * P**2
        + 0.25 * params['alpha_2'] * P**4
        + (1.0/6.0) * params['alpha_3'] * P**6
        + 0.5 * beta_1_eff * R**2
        + 0.25 * params['beta_2'] * R**4
        + (1.0/6.0) * params['beta_3'] * R**6
        + params['gamma_22'] * P**2 * R**2
        + params['gamma_42'] * P**4 * R**2
        + params['gamma_64'] * P**6 * R**4
        + params['gamma_84'] * P**8 * R**4
        + params['gamma_86'] * P**8 * R**6
        - P * E  # 電場Eを使用
    )

def find_stable_point(P, R, free_energy, init_p, init_r):
    """安定点探索"""
    idx_p = (np.abs(P[:,0] - init_p)).argmin()
    idx_r = (np.abs(R[0,:] - init_r)).argmin()
    shift_point = (idx_p, idx_r)

    for _ in range(10000):
        center_p, center_r = shift_point
        min_point = shift_point
        min_energy = free_energy[center_p, center_r]

        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1), ( 0, 0), ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]
        for dP, dR in neighbors:
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

def plot_2d_contour(P, R, free_energy, stable_p, stable_r, T, n, subdir, params, z_lim=None, param_text="", prev_stable_p=None, prev_stable_r=None, simple_mode=None):
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    # 品質モードではより大きな図サイズと高DPI
    figsize = (8, 7) if not SPEED_MODE else (6, 5)
    dpi = 300 if not SPEED_MODE else 100
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # エネルギーカットオフでマスク
    free_energy_masked = np.ma.masked_where(free_energy >= ENERGY_CUTOFF, free_energy)
    
    if z_lim:
        # 低エネルギー領域を強調するため、より細かいコンターを描画
        # 品質モードではより多くのレベルを使用
        levels = np.linspace(z_lim[0], z_lim[1], 80 if SPEED_MODE else 120)
    else:
        # マスクされた値以外の最小値・最大値でレベルを設定
        valid_energy = free_energy_masked[~free_energy_masked.mask] if np.any(~free_energy_masked.mask) else free_energy
        levels = np.linspace(np.min(valid_energy), np.max(valid_energy), 80 if SPEED_MODE else 120)

    # 低エネルギー領域の勾配をみやすくするため、より細かいコンターを描画
    # 品質モードではより太い線とアンチエイリアス
    contour = ax.contour(P, R, free_energy_masked, levels=levels, cmap="jet", 
                         linewidths=0.8 if SPEED_MODE else 1.2, antialiased=True)
    if is_show_contour_value:
        # 低エネルギー領域のコンターラベルをより多く表示
        step = 3 if SPEED_MODE else 5
        contour.clabel(contour.levels[::step], fmt='%.02f' if not SPEED_MODE else '%.01f', 
                      fontsize=5 if SPEED_MODE else 7)

    # 品質モードではより太い線と大きなマーカー
    trajectory_linewidth = 4 if SPEED_MODE else 5
    prev_markersize = 6 if SPEED_MODE else 8
    stable_markersize = 8 if SPEED_MODE else 10
    
    # 前の安定点から現在の安定点への軌跡を描画（SHOW_TRAJECTORYがTrueの場合、シンプルモードでも表示）
    if SHOW_TRAJECTORY and prev_stable_p is not None and prev_stable_r is not None:
        if not simple_mode:
            ax.plot([prev_stable_p, stable_p], [prev_stable_r, stable_r], 'r-', 
                    linewidth=trajectory_linewidth, alpha=0.6, label='Trajectory', antialiased=True)
        else:
            ax.plot([prev_stable_p, stable_p], [prev_stable_r, stable_r], 'r-', 
                    linewidth=trajectory_linewidth, alpha=0.6, antialiased=True)
        ax.plot(prev_stable_p, prev_stable_r, 'ro', markersize=prev_markersize, alpha=0.2)  # 前の点を薄い赤で表示
    
    ax.plot(stable_p, stable_r, 'ro', markersize=stable_markersize)
    # 安定点のPに基づく R^2, R^4, R^6 の係数値を計算して枠外に注記
    R2_val, R4_val, R6_val, total_val, discriminant_D = calc_r_coefficients(stable_p, T, params)
    
    if not simple_mode:
        ax.set_title(f"2D Free Energy Map at T={T:.3f}")
    # 軸ラベルを大きく太字で表示
    label_fontsize = 16 if not SPEED_MODE else 14
    ax.set_xlabel("P", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel("R", fontsize=label_fontsize, fontweight='bold')
    ax.set_xlim(contour_graph_x)
    ax.set_ylim(contour_graph_y)
    # 品質モードではより大きなフォントサイズ
    tick_labelsize = 9 if not SPEED_MODE else 8
    ax.tick_params(labelsize=tick_labelsize)
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
    # 品質モードでは高DPIで保存
    save_dpi = 300 if not SPEED_MODE else 100
    plt.savefig(os.path.join(subdir, f"2Dcontour_{n:04d}.png"), bbox_inches='tight', dpi=save_dpi)
    plt.close()

def plot_3d_surface(P, R, free_energy, stable_p, stable_r, T, n, subdir, params, z_lim=None, param_text="", simple_mode=None):
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    # 品質モードではより大きな図サイズと高DPI
    figsize = (10, 8) if not SPEED_MODE else (12, 6)
    dpi = 300 if not SPEED_MODE else 100
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # エネルギーカットオフでクリップ
    free_energy_smooth = np.where(free_energy >= ENERGY_CUTOFF, ENERGY_CUTOFF, free_energy)
    
    # 品質モードではアンチエイリアスを有効化し、より細かいメッシュ
    ax.plot_surface(P, R, free_energy_smooth, cmap=cm.jet, linewidth=0, 
                     antialiased=True if not SPEED_MODE else False, alpha=0.6)
    stable_energy = free_energy[(np.abs(P[:,0]-stable_p)).argmin(), (np.abs(R[0,:]-stable_r)).argmin()]
    # 品質モードではより大きなマーカー
    marker_size = 70 if not SPEED_MODE else 50
    ax.scatter(stable_p, stable_r, stable_energy, color='r', s=marker_size)
    # 安定点のPに基づく R^2, R^4, R^6 の係数値を計算して枠外に注記
    R2_val, R4_val, R6_val, total_val, discriminant_D = calc_r_coefficients(stable_p, T, params)
    
    if not simple_mode:
        ax.set_title(f"3D Free Energy Map at T={T:.3f}", pad=2)
    # 軸ラベルを大きく太字で表示
    label_fontsize = 16 if not SPEED_MODE else 14
    ax.set_xlabel("P", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel("R", fontsize=label_fontsize, fontweight='bold')
    ax.set_zlabel("F(P,R)", fontsize=label_fontsize, fontweight='bold', labelpad=10)  # ラベルの位置を調整して見切れを防ぐ
    # 品質モードではより大きなフォントサイズ
    tick_labelsize = 9 if not SPEED_MODE else 8
    ax.tick_params(labelsize=tick_labelsize)

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
    # 品質モードでは高DPIで保存
    save_dpi = 300 if not SPEED_MODE else 100
    plt.savefig(os.path.join(subdir, f"3Dsurf_{n:04d}.png"), bbox_inches='tight', pad_inches=0.1, dpi=save_dpi)
    plt.close()

def plot_cross_section_P(P, R, free_energy, stable_p, stable_r, T, n, subdir, params, z_lim=None, param_text="", simple_mode=None):
    """P軸に垂直な断面でのエネルギー最下点をプロット"""
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    # 品質モードではより大きな図サイズと高DPI
    figsize = (8, 6) if not SPEED_MODE else (6, 5)
    dpi = 300 if not SPEED_MODE else 100
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # エネルギー最下点を各P値で探索（ベクトル化）
    # マスクされた値や極端に大きな値を除外
    energy_masked = np.where(np.isnan(free_energy) | (free_energy >= 50000), np.inf, free_energy)
    min_energies = np.min(energy_masked, axis=1)
    # 全てinfの場合はnanに変換
    min_energies = np.where(np.isinf(min_energies), np.nan, min_energies)
    
    P_values = P[:, 0]
    
    # プロット（品質モードではより太い線）
    linewidth = 2.0 if not SPEED_MODE else 1.5
    if not simple_mode:
        ax.plot(P_values, min_energies, 'b-', linewidth=linewidth, label='Energy minimum', antialiased=True)
    else:
        ax.plot(P_values, min_energies, 'b-', linewidth=linewidth, antialiased=True)
    
    # stable_pに対応するエネルギー値を探す
    stable_idx = (np.abs(P_values - stable_p)).argmin()
    stable_energy = min_energies[stable_idx]
    # 品質モードではより大きなマーカー
    stable_markersize = 10 if SPEED_MODE else 12
    if not simple_mode:
        ax.plot(stable_p, stable_energy, 'ro', markersize=stable_markersize, 
                label=f'Stable P={stable_p:.3f}')
    else:
        ax.plot(stable_p, stable_energy, 'ro', markersize=stable_markersize)
    
    # 安定点のPに基づく R^2, R^4, R^6 の係数値を計算して枠外に注記
    R2_val, R4_val, R6_val, total_val, discriminant_D = calc_r_coefficients(stable_p, T, params)
    
    if not simple_mode:
        ax.set_title(f"Cross-section Energy Min (R-axis) at T={T:.3f}")
    # 軸ラベルを大きく太字で表示
    label_fontsize = 16 if not SPEED_MODE else 14
    ax.set_xlabel("P", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel("Minimum Energy", fontsize=label_fontsize, fontweight='bold')
    ax.set_xlim(contour_graph_x)
    if z_lim:
        ax.set_ylim(z_lim)
    # 品質モードではより大きなフォントサイズ
    tick_labelsize = 9 if not SPEED_MODE else 8
    ax.tick_params(labelsize=tick_labelsize)
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
    # 品質モードでは高DPIで保存
    save_dpi = 300 if not SPEED_MODE else 100
    plt.savefig(os.path.join(subdir, f"cross_section_P_{n:04d}.png"), bbox_inches='tight', dpi=save_dpi)
    plt.close()

def plot_cross_section_R(P, R, free_energy, stable_p, stable_r, T, n, subdir, params, z_lim=None, param_text="", simple_mode=None):
    """R軸に垂直な断面でのエネルギー最下点をプロット"""
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    # 品質モードではより大きな図サイズと高DPI
    figsize = (8, 6) if not SPEED_MODE else (6, 5)
    dpi = 300 if not SPEED_MODE else 100
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # エネルギー最下点を各R値で探索（ベクトル化）
    # マスクされた値や極端に大きな値を除外
    energy_masked = np.where(np.isnan(free_energy) | (free_energy >= 50000), np.inf, free_energy)
    min_energies = np.min(energy_masked, axis=0)
    # 全てinfの場合はnanに変換
    min_energies = np.where(np.isinf(min_energies), np.nan, min_energies)
    
    R_values = R[0, :]
    
    # プロット（品質モードではより太い線）
    linewidth = 2.0 if not SPEED_MODE else 1.5
    if not simple_mode:
        ax.plot(R_values, min_energies, 'b-', linewidth=linewidth, label='Energy minimum', antialiased=True)
    else:
        ax.plot(R_values, min_energies, 'b-', linewidth=linewidth, antialiased=True)
    
    # stable_rに対応するエネルギー値を探す
    stable_idx = (np.abs(R_values - stable_r)).argmin()
    stable_energy = min_energies[stable_idx]
    # 品質モードではより大きなマーカー
    stable_markersize = 10 if SPEED_MODE else 12
    if not simple_mode:
        ax.plot(stable_r, stable_energy, 'ro', markersize=stable_markersize, 
                label=f'Stable R={stable_r:.3f}')
    else:
        ax.plot(stable_r, stable_energy, 'ro', markersize=stable_markersize)
    
    # 安定点のPに基づく R^2, R^4, R^6 の係数値を計算して枠外に注記
    R2_val, R4_val, R6_val, total_val, discriminant_D = calc_r_coefficients(stable_p, T, params)
    
    if not simple_mode:
        ax.set_title(f"Cross-section Energy Min (P-axis) at T={T:.3f}")
    # 軸ラベルを大きく太字で表示
    label_fontsize = 16 if not SPEED_MODE else 14
    ax.set_xlabel("R", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel("Minimum Energy", fontsize=label_fontsize, fontweight='bold')
    ax.set_xlim(contour_graph_y)
    if z_lim:
        ax.set_ylim(z_lim)
    # 品質モードではより大きなフォントサイズ
    tick_labelsize = 9 if not SPEED_MODE else 8
    ax.tick_params(labelsize=tick_labelsize)
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
    # 品質モードでは高DPIで保存
    save_dpi = 300 if not SPEED_MODE else 100
    plt.savefig(os.path.join(subdir, f"cross_section_R_{n:04d}.png"), bbox_inches='tight', dpi=save_dpi)
    plt.close()

def plot_hysteresis_P_T_frame(T_array, P_stable_array, T_current, P_current, current_idx, n, subdir, params, param_text="", simple_mode=None):
    """P-Tヒステリシスループの各フレームを生成（現在の点をハイライト）"""
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    # 品質モードではより大きな図サイズと高DPI
    figsize = (10, 7) if not SPEED_MODE else (8, 6)
    dpi = 300 if not SPEED_MODE else 100
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 品質モードではより太い線と大きなマーカー
    markersize_unplotted = 2 if SPEED_MODE else 3
    markersize_plotted = 3 if SPEED_MODE else 4
    linewidth_plotted = 2 if SPEED_MODE else 2.5
    markersize_current = 12 if SPEED_MODE else 14
    markersize_current_outline = 8 if SPEED_MODE else 10
    markeredgewidth = 2 if SPEED_MODE else 2.5
    
    # 未プロット部分（現在の点以降）を薄く描画（不透明度30%）
    if current_idx >= 0 and current_idx < len(T_array) - 1:
        ax.plot(T_array[current_idx+1:], P_stable_array[current_idx+1:], 'o-', 
                markersize=markersize_unplotted, linewidth=1, 
                color='blue', alpha=0.3, antialiased=True)
    
    # プロット済み部分（現在の点まで）を通常の色で描画
    if not simple_mode:
        if current_idx >= 0 and current_idx < len(T_array):
            ax.plot(T_array[:current_idx+1], P_stable_array[:current_idx+1], 
                    'o-', markersize=markersize_plotted, linewidth=linewidth_plotted, 
                    color='blue', alpha=1.0, label="P(T) path", antialiased=True)
        
        # 現在の点をハイライト（大きく赤い点）
        ax.plot(T_current, P_current, 'ro', markersize=markersize_current, 
                label=f'Current T={T_current:.3f}')
        ax.plot(T_current, P_current, 'ro', markersize=markersize_current_outline, 
                markerfacecolor='none', markeredgewidth=markeredgewidth)
    else:
        if current_idx >= 0 and current_idx < len(T_array):
            ax.plot(T_array[:current_idx+1], P_stable_array[:current_idx+1], 
                    'o-', markersize=markersize_plotted, linewidth=linewidth_plotted, 
                    color='blue', alpha=1.0, antialiased=True)
        
        # 現在の点をハイライト（大きく赤い点）
        ax.plot(T_current, P_current, 'ro', markersize=markersize_current)
        ax.plot(T_current, P_current, 'ro', markersize=markersize_current_outline, 
                markerfacecolor='none', markeredgewidth=markeredgewidth)
    
    if not simple_mode:
        ax.set_title(f"P vs Temperature (Hysteresis Loop) at T={T_current:.3f}")
    # 軸ラベルを大きく太字で表示
    label_fontsize = 16 if not SPEED_MODE else 14
    ax.set_xlabel("T", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel("P", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylim(P_T_graph_y)
    # 品質モードではより大きなフォントサイズ
    tick_labelsize = 9 if not SPEED_MODE else 8
    ax.tick_params(labelsize=tick_labelsize)
    ax.grid(True, alpha=0.3)  # シンプルモードでもグリッドを表示
    if not simple_mode:
        ax.legend(loc='best', fontsize=8)
        
        ax.text(1.05, 0.95, param_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
    
    # ディレクトリはメインループで既に作成済み
    # 品質モードでは高DPIで保存
    save_dpi = 300 if not SPEED_MODE else 100
    plt.savefig(os.path.join(subdir, f"hysteresis_P_T_{n:04d}.png"), bbox_inches='tight', dpi=save_dpi)
    plt.close()

def plot_hysteresis_R_T_frame(T_array, R_stable_array, T_current, R_current, current_idx, n, subdir, params, param_text="", simple_mode=None):
    """R-Tヒステリシスループの各フレームを生成（現在の点をハイライト）"""
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    # 品質モードではより大きな図サイズと高DPI
    figsize = (10, 7) if not SPEED_MODE else (8, 6)
    dpi = 300 if not SPEED_MODE else 100
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 品質モードではより太い線と大きなマーカー
    markersize_unplotted = 2 if SPEED_MODE else 3
    markersize_plotted = 3 if SPEED_MODE else 4
    linewidth_plotted = 2 if SPEED_MODE else 2.5
    markersize_current = 12 if SPEED_MODE else 14
    markersize_current_outline = 8 if SPEED_MODE else 10
    markeredgewidth = 2 if SPEED_MODE else 2.5
    
    # 未プロット部分（現在の点以降）を薄く描画（不透明度30%）
    if current_idx >= 0 and current_idx < len(T_array) - 1:
        ax.plot(T_array[current_idx+1:], R_stable_array[current_idx+1:], 'o-', 
                markersize=markersize_unplotted, linewidth=1, 
                color='green', alpha=0.3, antialiased=True)
    
    # プロット済み部分（現在の点まで）を通常の色で描画
    if not simple_mode:
        if current_idx >= 0 and current_idx < len(T_array):
            ax.plot(T_array[:current_idx+1], R_stable_array[:current_idx+1], 
                    'o-', markersize=markersize_plotted, linewidth=linewidth_plotted, 
                    color='green', alpha=1.0, label="R(T) path", antialiased=True)
        
        # 現在の点をハイライト（大きく赤い点）
        ax.plot(T_current, R_current, 'ro', markersize=markersize_current, 
                label=f'Current T={T_current:.3f}')
        ax.plot(T_current, R_current, 'ro', markersize=markersize_current_outline, 
                markerfacecolor='none', markeredgewidth=markeredgewidth)
    else:
        if current_idx >= 0 and current_idx < len(T_array):
            ax.plot(T_array[:current_idx+1], R_stable_array[:current_idx+1], 
                    'o-', markersize=markersize_plotted, linewidth=linewidth_plotted, 
                    color='green', alpha=1.0, antialiased=True)
        
        # 現在の点をハイライト（大きく赤い点）
        ax.plot(T_current, R_current, 'ro', markersize=markersize_current)
        ax.plot(T_current, R_current, 'ro', markersize=markersize_current_outline, 
                markerfacecolor='none', markeredgewidth=markeredgewidth)
    
    if not simple_mode:
        ax.set_title(f"R vs Temperature (Hysteresis Loop) at T={T_current:.3f}")
    # 軸ラベルを大きく太字で表示
    label_fontsize = 16 if not SPEED_MODE else 14
    ax.set_xlabel("T", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel("R", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylim(R_T_graph_y)
    # 品質モードではより大きなフォントサイズ
    tick_labelsize = 9 if not SPEED_MODE else 8
    ax.tick_params(labelsize=tick_labelsize)
    ax.grid(True, alpha=0.3)  # シンプルモードでもグリッドを表示
    if not simple_mode:
        ax.legend(loc='best', fontsize=8)
        
        ax.text(1.05, 0.95, param_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
    
    # ディレクトリはメインループで既に作成済み
    # 品質モードでは高DPIで保存
    save_dpi = 300 if not SPEED_MODE else 100
    plt.savefig(os.path.join(subdir, f"hysteresis_R_T_{n:04d}.png"), bbox_inches='tight', dpi=save_dpi)
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
                              P, R, T, n, subdir, param_text="", 
                              ti_positions_original=None, o_positions_original=None,
                              ti_positions_prev=None, o_positions_prev=None, simple_mode=None):
    """
    視点1：1ユニットセルを斜めから見た3Dモデル
    """
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    # 品質モードではより大きな図サイズと高DPI
    figsize = (12, 10) if not SPEED_MODE else (10, 8)
    dpi = 300 if not SPEED_MODE else 100
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # 原子のサイズ（Aサイト、Bサイトを大きく、2.5倍に拡大）
    ba_size = 3500
    ti_size = 2500
    o_size = 1000
    
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
        ax.set_title(f'BaTiO3 Unit Cell (Oblique View)\nP={P:.3f}, R={R:.3f}, T={T:.3f}')
        ax.legend(loc='upper right', fontsize=8, markerscale=0.6, framealpha=0.9, fancybox=True)
    
    # アスペクト比を1:1:1に設定（立方晶を正しく表示）
    # データ範囲を取得してアスペクト比を設定
    all_positions = np.vstack([ba_positions, ti_positions, o_positions])
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
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
    
    # シンプルモードの時は軸範囲をギリギリに設定（余白なし）
    if simple_mode:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        # 軸の余白を0に設定
        ax.margins(0)
        # 図の余白を最小化
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # PNG保存時に背景を白に設定し、透明度を正しく保存
    # 3Dプロットの透明度を正しくラスタライズするため、一度描画してから保存
    plt.draw()
    
    # シンプルモードの時は描画後に再度範囲を設定して余白を完全に削除
    if simple_mode:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.margins(0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.draw()
    
    # ディレクトリはメインループで既に作成済み
    # 品質モードでは高DPIで保存
    save_dpi = 300 if not SPEED_MODE else 150
    # シンプルモードの時は余白をなくす
    pad_inches_value = 0 if simple_mode else 0.1
    plt.savefig(os.path.join(subdir, f"3Dunitcell_oblique_{n:04d}.png"), 
                bbox_inches='tight', dpi=save_dpi, facecolor='white', edgecolor='none', 
                transparent=False, pad_inches=pad_inches_value)
    plt.close()

def plot_3d_unitcell_topview(ba_positions, ti_positions, o_positions, octahedra_info,
                               P, R, T, n, subdir, param_text="", o_positions_original=None,
                               ti_positions_prev=None, o_positions_prev=None, simple_mode=None):
    """
    【軽量化版】視点2：2×2×1ユニットセルをC軸垂直方向から見た2D投影
    3D処理を廃止し、純粋な2D描画を行うことで高速化・省メモリ化を実現
    """
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
        
    # 3Dではなく通常の2Dプロットを作成（これで劇的に軽くなります）
    # 品質モードではより大きな図サイズと高DPI
    figsize = (18, 12) if not SPEED_MODE else (16, 10)
    dpi = 300 if not SPEED_MODE else 100
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 原子のサイズ（2D用、2.5倍に拡大）
    ba_size = 1250
    ti_size = 1250
    o_size = 750
    
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
        ax.set_title(f'BaTiO3 Supercell (2×2×1) Top View [2D Mode]\nP={P:.3f}, R={R:.3f}, T={T:.3f}')
        ax.legend(loc='upper right', fontsize=8, markerscale=0.6, framealpha=0.9, fancybox=True)
    
    ax.set_aspect('equal')
    
    # 表示範囲の設定
    all_x = np.concatenate([ba_positions[:, 0], ti_positions[:, 0], o_positions[:, 0]])
    all_y = np.concatenate([ba_positions[:, 1], ti_positions[:, 1], o_positions[:, 1]])
    
    # シンプルモードの時はマージンを少し確保（余白あり）
    # 見切れを防ぐため、適度なマージンを確保
    margin = 0.15 if simple_mode else 0.5
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    # シンプルモードの時は軸の余白を最小化（ただしsubplots_adjustは使わない）
    if simple_mode:
        ax.margins(0)
    
    ax.axis('off')



    # パラメータテキスト
    if not simple_mode:
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))



    # 保存（ディレクトリはメインループで既に作成済み）
    # 品質モードでは高DPIで保存
    save_dpi = 300 if not SPEED_MODE else 150
    # シンプルモードの時は余白をなくす
    pad_inches_value = 0 if simple_mode else 0.1
    plt.savefig(os.path.join(subdir, f"3Dunitcell_topview_{n:04d}.png"), 
                bbox_inches='tight', dpi=save_dpi, pad_inches=pad_inches_value)
    plt.close()

def create_animation(root_folder, output_filename, pattern, start_index=0, pbar=None, total_duration=None):
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
    # total_durationが指定されている場合、フレーム数に応じてdurationを計算
    if total_duration is not None:
        duration = total_duration / len(full_path_list)
    else:
        duration = 0.1 if SPEED_MODE else 0.2
    fps = 10 if SPEED_MODE else 15  # スピードモード: 10fps, 品質モード: 15fps
    
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
        # writerの作成パラメータを条件分岐
        writer_kwargs = {'mode': 'I', 'duration': duration}
        if fps:
            writer_kwargs['fps'] = fps
        
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
        patterns.append(("hysteresis_P_T_", "P-T Hysteresis"))
        patterns.append(("hysteresis_R_T_", "R-T Hysteresis"))
    if GENERATE_3D_UNITCELL:
        patterns.append(("3Dunitcell_oblique_", "3D Unit Cell Oblique"))
        patterns.append(("3Dunitcell_topview_", "3D Unit Cell Topview"))
    
    if not patterns:
        tqdm.write("No plot types enabled for summary GIF")
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
        tqdm.write("No images found for summary GIF")
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
    duration = 0.1 if SPEED_MODE else 0.2
    fps = 10 if SPEED_MODE else 15
    
    try:
        if fps:
            writer = imageio.get_writer(output_path, mode='I', duration=duration, fps=fps)
        else:
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
            
            if (frame_idx + 1) % 10 == 0:
                tqdm.write(f"Processed {frame_idx + 1}/{max_frames} frames for summary GIF...")
        
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
    
    # temp_T_speed/temp_T_qualityフォルダのみをリセット
    if os.path.isdir(GRAPH_SAVE_DIR):
        shutil.rmtree(GRAPH_SAVE_DIR)
    os.makedirs(GRAPH_SAVE_DIR)

    P_values = np.arange(P_grid[0], P_grid[1] + grid_mesh, grid_mesh)
    R_values = np.arange(R_grid[0], R_grid[1] + grid_mesh, grid_mesh)
    P_mesh, R_mesh = np.meshgrid(P_values, R_values, indexing='ij')

    print(f"Running in {'SPEED MODE' if SPEED_MODE else 'QUALITY MODE'}")
    print(f"GENERATE_3D_UNITCELL = {GENERATE_3D_UNITCELL}")
    
    # ターミナル出力のバッファリングを無効化（プログレスバーが確実に表示されるように）
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except:
            pass
    
    # 電場範囲の設定
    # 1回目のループ用：0→E_max→E_min→0（2nd_P-R.pyと同じパターン）
    E_range_1st = np.concatenate([
        np.linspace(0, E_max, E_divisions + 1)[:-1],  # 0からE_maxまで（E_maxは含まない）
        np.linspace(E_max, E_min, E_divisions + 1)[:-1],  # E_maxからE_minまで（E_minは含まない）
        np.linspace(E_min, 0, E_divisions + 1)  # E_minから0まで（0は含む）
    ])
    
    # 2回目のループ用：0→E_max→0（DCバイアス測定の準備、positiveの分極状態から開始）
    E_range_2nd = np.concatenate([
        np.linspace(0, E_max, E_divisions + 1)[:-1],  # 0からE_maxまで（E_maxは含まない）
        np.linspace(E_max, 0, E_divisions + 1)  # E_maxから0まで（0は含む）
    ])
    
    # 2回目のループ用（P-Eヒステリシスループ描画用）：0→E_max→E_min→0（参照ファイルと同じパターン）
    E_range_2nd_hysteresis = np.concatenate([
        np.linspace(0, E_max, E_divisions + 1)[:-1],  # 0からE_maxまで（E_maxは含まない）
        np.linspace(E_max, E_min, E_divisions + 1)[:-1],  # E_maxからE_minまで（E_minは含まない）
        np.linspace(E_min, 0, E_divisions + 1)  # E_minから0まで（0は含む）
    ])
    
    # 3回目のループ用：DCバイアス測定（0→E_max）
    E_values_dc = np.arange(0, E_max + E_step, E_step)
    
    tqdm.write(f"DC Bias Dielectric Measurement Mode")
    tqdm.write(f"Electric field range (absolute): ±{E_range_abs} (divisions: {E_divisions})")
    tqdm.write(f"DC bias measurement range: 0 → {E_max} (step: {E_step})")
    tqdm.write(f"Fixed temperature: T = {fixed_params['T_FIXED']}")

    # 変数名を下付き文字で表示するための変換
    def format_param_name(param_name):
        # 下付き文字の変換辞書
        subscript_map = {
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
            'gamma_64': 'γ_{64}',
            'gamma_84': 'γ_{84}',
            'gamma_86': 'γ_{86}'
        }
        return subscript_map.get(param_name, param_name)
    
    # format_value_with_colorを使用（重複関数を削除）
    # γ項は変数値が0の時は表示しない、α・β項は0でも表示する
    param_text_for_plots = "\n".join([
        f"${format_param_name(k)}$ = ${format_value_with_color(v)}$" 
        for k, v in fixed_params.items() 
        if not (k.startswith('gamma_') and v == 0)
    ])

    # ==================================================================
    # 【1回目のループ】電場0→E_max→E_min→0で計算（P-Eヒステリシスループ用データ取得）
    # ==================================================================
    tqdm.write("=" * 60)
    tqdm.write("1st Loop: Electric field sweep 0 → E_max → E_min → 0 (hysteresis loop data)")
    tqdm.write("=" * 60)
    
    current_p = init_P_value
    current_r = init_R_value
    
    # 固定温度を使用
    T_fixed = fixed_params['T_FIXED']
    
    # 1回目のループ用のデータを保存（初期化用）
    for E in tqdm(E_range_1st, desc="1st E sweep", leave=False):
        # 自由エネルギーを計算
        free_energy = calc_free_energy(P_mesh, R_mesh, T_fixed, E, fixed_params)
        # エネルギーカットオフ
        free_energy = np.where(free_energy > ENERGY_CUTOFF, ENERGY_CUTOFF, free_energy)
        
        # 安定点を探索
        stable_p, stable_r = find_stable_point(P_mesh, R_mesh, free_energy, current_p, current_r)
        
        # NaN/Infチェック
        if not np.isfinite(stable_p) or not np.isfinite(stable_r):
            stable_p = current_p if np.isfinite(current_p) else 0.0
            stable_r = current_r if np.isfinite(current_r) else 0.0
        
        current_p = stable_p
        current_r = stable_r
    
    tqdm.write(f"1st loop completed. Final P={current_p:.3f}, R={current_r:.3f}")
    
    # ==================================================================
    # 【2回目のループ】P-Eヒステリシスループ描画用：0→E_max→E_min→0（参照ファイルと同じパターン）
    # ==================================================================
    tqdm.write("=" * 60)
    tqdm.write("2nd Loop: Electric field sweep 0 → E_max → E_min → 0 (for P-E hysteresis loop plot)")
    tqdm.write("=" * 60)
    
    # 1回目のループの最終状態から開始（参照ファイルと同じ、positiveの分極状態にする処理は不要）
    current_p_hysteresis = current_p
    current_r_hysteresis = current_r
    
    # 2回目のループ用のデータを保存（P-Eヒステリシスループ用）
    E_hysteresis = []  # 電場値（0→E_max→E_min→0）
    P_hysteresis = []  # 分極値（0→E_max→E_min→0）
    
    for E in tqdm(E_range_2nd_hysteresis, desc="2nd E sweep (hysteresis)", leave=False):
        # 自由エネルギーを計算
        free_energy = calc_free_energy(P_mesh, R_mesh, T_fixed, E, fixed_params)
        # エネルギーカットオフ
        free_energy = np.where(free_energy > ENERGY_CUTOFF, ENERGY_CUTOFF, free_energy)
        
        # 安定点を探索
        stable_p, stable_r = find_stable_point(P_mesh, R_mesh, free_energy, current_p_hysteresis, current_r_hysteresis)
        
        # NaN/Infチェック
        if not np.isfinite(stable_p) or not np.isfinite(stable_r):
            stable_p = current_p_hysteresis if np.isfinite(current_p_hysteresis) else 0.0
            stable_r = current_r_hysteresis if np.isfinite(current_r_hysteresis) else 0.0
        
        # ヒステリシスループ用データを保存
        E_hysteresis.append(E)
        P_hysteresis.append(stable_p)
        
        current_p_hysteresis = stable_p
        current_r_hysteresis = stable_r
    
    tqdm.write(f"2nd loop (hysteresis) completed. Final P={current_p_hysteresis:.3f}, R={current_r_hysteresis:.3f}")
    
    # デバッグ用：データの範囲を確認
    if len(E_hysteresis) > 0 and len(P_hysteresis) > 0:
        E_hyst_arr = np.array(E_hysteresis)
        P_hyst_arr = np.array(P_hysteresis)
        tqdm.write(f"E range: [{E_hyst_arr.min():.1f}, {E_hyst_arr.max():.1f}]")
        tqdm.write(f"P range: [{P_hyst_arr.min():.3f}, {P_hyst_arr.max():.3f}]")
        tqdm.write(f"Number of negative P values: {np.sum(P_hyst_arr < 0)}")
        tqdm.write(f"Number of positive P values: {np.sum(P_hyst_arr > 0)}")
        # Eが負の範囲でのPの範囲を確認
        negative_E_mask = E_hyst_arr < 0
        if np.any(negative_E_mask):
            tqdm.write(f"P range for E < 0: [{P_hyst_arr[negative_E_mask].min():.3f}, {P_hyst_arr[negative_E_mask].max():.3f}]")
    
    # ==================================================================
    # 【P-Eヒステリシスループのプロット（2回目のループ結果）- 通常版】
    # ==================================================================
    tqdm.write("=" * 60)
    tqdm.write("Plotting P-E hysteresis loop (2nd loop results) - Normal version")
    tqdm.write("=" * 60)
    
    if len(E_hysteresis) > 0 and len(P_hysteresis) > 0:
        # 通常のP-Eヒステリシスループのプロット（ハイライトなし）
        figsize = (10, 7) if not SPEED_MODE else (8, 6)
        dpi = 300 if not SPEED_MODE else 100
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        markersize = 4 if not SPEED_MODE else 3
        linewidth = 2.5 if not SPEED_MODE else 2.0
        
        # ヒステリシスループ全体を描画（ハイライトなし）
        if not SIMPLE_PLOT_MODE:
            ax.plot(E_hysteresis, P_hysteresis, 'o-', markersize=markersize,
                    linewidth=linewidth, label="2nd: P(E)", 
                    color='blue', antialiased=True)
            ax.set_title("P vs Electric Field (2nd cycle)")
            ax.legend()
            ax.text(1.05, 0.95, param_text_for_plots, transform=ax.transAxes,
                    fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
        else:
            ax.plot(E_hysteresis, P_hysteresis, 'o-', markersize=markersize,
                    linewidth=linewidth, color='blue', antialiased=True)
        
        # 軸ラベルをシンプルに表示（2nd_P-R.pyと同じスタイル）
        ax.set_xlabel("E")
        ax.set_ylabel("P")
        ax.set_ylim(P_E_graph_y)
        ax.grid()
        
        save_dpi = 300 if not SPEED_MODE else 100
        plt.savefig(os.path.join(GRAPH_SAVE_DIR, "P_E_hysteresis_loop.png"), 
                   bbox_inches='tight', dpi=save_dpi)
        plt.close()
        
        tqdm.write(f"Normal P-E hysteresis loop plot saved: {len(E_hysteresis)} data points")
    else:
        tqdm.write("Warning: No valid P-E hysteresis data to plot")
    
    # ==================================================================
    # 【DCバイアス測定の準備】0→E_max→0で電場印加を行い、positiveの分極で安定させる
    # ==================================================================
    tqdm.write("=" * 60)
    tqdm.write("DC Bias Preparation: Electric field sweep 0 → E_max → 0 (to stabilize positive polarization)")
    tqdm.write("=" * 60)
    
    # 1回目のループの最終状態から開始
    current_p = current_p_hysteresis
    current_r = current_r_hysteresis
    
    # positiveの分極状態を確保するため、E_maxを印加してpositiveの分極状態にする
    E_positive = E_max
    free_energy_positive = calc_free_energy(P_mesh, R_mesh, T_fixed, E_positive, fixed_params)
    free_energy_positive = np.where(free_energy_positive > ENERGY_CUTOFF, ENERGY_CUTOFF, free_energy_positive)
    stable_p_positive, stable_r_positive = find_stable_point(P_mesh, R_mesh, free_energy_positive, current_p, current_r)
    
    # NaN/Infチェック
    if not np.isfinite(stable_p_positive) or not np.isfinite(stable_r_positive):
        stable_p_positive = current_p if np.isfinite(current_p) else 0.0
        stable_r_positive = current_r if np.isfinite(current_r) else 0.0
    
    # positiveの分極状態から開始
    current_p = stable_p_positive
    current_r = stable_r_positive
    
    # 0→E_max→0の電場印加
    for E in tqdm(E_range_2nd, desc="DC bias prep (0→E_max→0)", leave=False):
        # 自由エネルギーを計算
        free_energy = calc_free_energy(P_mesh, R_mesh, T_fixed, E, fixed_params)
        # エネルギーカットオフ
        free_energy = np.where(free_energy > ENERGY_CUTOFF, ENERGY_CUTOFF, free_energy)
        
        # 安定点を探索
        stable_p, stable_r = find_stable_point(P_mesh, R_mesh, free_energy, current_p, current_r)
        
        # NaN/Infチェック
        if not np.isfinite(stable_p) or not np.isfinite(stable_r):
            stable_p = current_p if np.isfinite(current_p) else 0.0
            stable_r = current_r if np.isfinite(current_r) else 0.0
        
        current_p = stable_p
        current_r = stable_r
    
    tqdm.write(f"DC bias preparation completed. Final P={current_p:.3f}, R={current_r:.3f}")
    
    # ==================================================================
    # 【3回目のループ】0→E_maxの範囲でDCバイアス測定（誘電率計算）
    # ==================================================================
    tqdm.write("=" * 60)
    tqdm.write("3rd Loop: DC bias measurement 0 → E_max (dielectric constant calculation)")
    tqdm.write("=" * 60)
    
    # 誘電率データを保存
    E_array = []  # 電場値
    P_array = []  # 分極値
    
    total_loops = len(E_values_dc)
    main_pbar = tqdm(total=total_loops, desc="3rd loop (DC bias)", 
                     mininterval=0.1, maxinterval=1.0, leave=True)
    
    # 2回目のループの最終値から開始（positiveの分極状態）
    temp_p = current_p
    temp_r = current_r
    
    for E in E_values_dc:
        # 自由エネルギーを計算（固定温度T_fixedで）
        free_energy = calc_free_energy(P_mesh, R_mesh, T_fixed, E, fixed_params)
        # エネルギーカットオフ
        free_energy = np.where(free_energy > ENERGY_CUTOFF, ENERGY_CUTOFF, free_energy)
        
        # 安定点を探索
        stable_p, stable_r = find_stable_point(P_mesh, R_mesh, free_energy, temp_p, temp_r)
        
        # NaN/Infチェック
        if not np.isfinite(stable_p) or not np.isfinite(stable_r):
            stable_p = temp_p if np.isfinite(temp_p) else 0.0
            stable_r = temp_r if np.isfinite(temp_r) else 0.0
        
        E_array.append(E)
        P_array.append(stable_p)
        
        temp_p = stable_p
        temp_r = stable_r
        main_pbar.update(1)
    
    main_pbar.close()
    
    # P-Eの傾き（dP/dE）を計算し、被誘電率ε_r = 1 + χを計算
    # 単位: P [μC/cm²], E [kV/cm]
    # 単位変換係数:
    #   - P: 1 μC/cm² = 10^-6 C / (10^-2 m)² = 10^-6 C / (10^-4 m²) = 10^-2 C/m²
    #   - E: 1 kV/cm = 10^3 V / (10^-2 m) = 10^5 V/m
    # dP/dE (SI単位): [C/m²] / [V/m] = C/(V·m) = F/m
    # 真空の誘電率: ε_0 = 8.854×10^-12 F/m
    # 無次元化: χ = (dP/dE) / ε_0
    # 被誘電率: ε_r = 1 + χ
    
    # 単位変換係数
    # P: μC/cm² → C/m² の変換係数
    #   1 μC/cm² = 10^-6 C / (10^-2 m)² = 10^-6 / 10^-4 C/m² = 10^-2 C/m²
    P_CONVERSION_FACTOR = 1e-2  # μC/cm² → C/m²
    
    # E: kV/cm → V/m の変換係数
    #   1 kV/cm = 10^3 V / (10^-2 m) = 10^5 V/m
    E_CONVERSION_FACTOR = 1e5  # kV/cm → V/m
    
    # 真空の誘電率 [F/m]
    EPSILON_0 = 8.854e-12  # F/m
    
    E_dielectric = []
    epsilon_r_array = []  # 被誘電率ε_r
    
    if len(E_array) > 1 and len(P_array) > 1:
        # 線形回帰で傾きを計算
        E_arr = np.array(E_array)
        P_arr = np.array(P_array)
        
        # 有効なデータのみを使用（NaN/Infを除外）
        valid_mask = np.isfinite(E_arr) & np.isfinite(P_arr)
        if np.sum(valid_mask) >= 2:
            E_valid = E_arr[valid_mask]
            P_valid = P_arr[valid_mask]
            
            # SI単位系に変換
            E_valid_SI = E_valid * E_CONVERSION_FACTOR  # V/m
            P_valid_SI = P_valid * P_CONVERSION_FACTOR  # C/m²
            
            # 線形回帰：P = a*E + b の傾きaを計算（SI単位系でのdP/dE [F/m]）
            if len(E_valid_SI) > 1 and np.var(E_valid_SI) > 1e-10:
                slope_SI = np.cov(E_valid_SI, P_valid_SI)[0, 1] / np.var(E_valid_SI)  # F/m
            else:
                # 差分で近似
                slope_SI = np.gradient(P_valid_SI, E_valid_SI).mean()  # F/m
            
            # 無次元の電気感受率を計算
            chi_dimensionless = slope_SI / EPSILON_0  # 無次元
            epsilon_r_overall = 1.0 + chi_dimensionless
            
            # 各電場での局所的な傾きを計算（5点間の傾きから算出）
            for i in range(len(E_array)):
                E_dielectric.append(E_array[i])
                
                # 5点間の傾きを計算（中心点iを含む前後2点ずつ、合計5点）
                # 利用可能な範囲を決定
                start_idx = max(0, i - 2)
                end_idx = min(len(E_array), i + 3)  # i+2まで含めるため+3
                
                # 5点のデータを抽出
                E_window = E_array[start_idx:end_idx]
                P_window = P_array[start_idx:end_idx]
                
                # 有効なデータのみを使用（NaN/Infを除外）
                valid_mask_window = np.isfinite(E_window) & np.isfinite(P_window)
                if np.sum(valid_mask_window) >= 2:
                    E_valid_window = np.array(E_window)[valid_mask_window]
                    P_valid_window = np.array(P_window)[valid_mask_window]
                    
                    # SI単位系に変換
                    E_valid_window_SI = E_valid_window * E_CONVERSION_FACTOR  # V/m
                    P_valid_window_SI = P_valid_window * P_CONVERSION_FACTOR  # C/m²
                    
                    # 線形回帰で傾きを計算（最小二乗法）
                    if len(E_valid_window_SI) > 1 and np.var(E_valid_window_SI) > 1e-10:
                        # 共分散と分散から傾きを計算
                        local_slope_SI = np.cov(E_valid_window_SI, P_valid_window_SI)[0, 1] / np.var(E_valid_window_SI)  # F/m
                    else:
                        # 差分で近似（データが少ない場合）
                        if len(E_valid_window_SI) >= 2:
                            local_slope_SI = np.gradient(P_valid_window_SI, E_valid_window_SI).mean()  # F/m
                        else:
                            # データが不足している場合は全体の平均値を使用
                            local_slope_SI = slope_SI
                    
                    # 無次元化
                    chi_local = local_slope_SI / EPSILON_0  # 無次元
                    epsilon_r = 1.0 + chi_local
                else:
                    # 有効なデータが不足している場合は全体の平均値を使用
                    epsilon_r = epsilon_r_overall
                
                epsilon_r_array.append(epsilon_r)
            
            tqdm.write(f"Overall susceptibility χ (dimensionless) = {chi_dimensionless:.6f}")
            tqdm.write(f"Overall relative permittivity ε_r = {epsilon_r_overall:.6f}")
        else:
            tqdm.write("Warning: Insufficient valid data for dielectric constant calculation")
    
    # ==================================================================
    # 【結果のプロット】被誘電率ε_r vs電場のグラフ
    # ==================================================================
    tqdm.write("=" * 60)
    tqdm.write("Plotting relative permittivity ε_r vs electric field")
    tqdm.write("=" * 60)
    
    if len(E_dielectric) > 0 and len(epsilon_r_array) > 0:
        # 被誘電率ε_r vs電場のグラフ
        figsize = (10, 7) if not SPEED_MODE else (8, 6)
        dpi = 300 if not SPEED_MODE else 100
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        markersize = 6 if not SPEED_MODE else 4
        linewidth = 2.5 if not SPEED_MODE else 2.0
        
        if not SIMPLE_PLOT_MODE:
            ax.plot(E_dielectric, epsilon_r_array, 'o-', markersize=markersize,
                    linewidth=linewidth, label="Relative Permittivity (ε_r)", 
                    color='blue', antialiased=True)
            ax.set_title("Relative Permittivity (ε_r) vs DC Bias Electric Field")
            ax.legend()
            ax.text(1.05, 0.95, param_text_for_plots, transform=ax.transAxes,
                    fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
        else:
            ax.plot(E_dielectric, epsilon_r_array, 'o-', markersize=markersize,
                    linewidth=linewidth, color='blue', antialiased=True)
        
        # 軸ラベルを大きく太字で表示
        label_fontsize = 16 if not SPEED_MODE else 14
        ax.set_xlabel("Electric Field E [kV/cm]", fontsize=label_fontsize, fontweight='bold')
        ax.set_ylabel("Relative Permittivity (ε_r)", fontsize=label_fontsize, fontweight='bold')
        ax.grid()
        
        save_dpi = 300 if not SPEED_MODE else 100
        plt.savefig(os.path.join(GRAPH_SAVE_DIR, "relative_permittivity_vs_E.png"), 
                   bbox_inches='tight', dpi=save_dpi)
        plt.close()
        
        tqdm.write(f"Relative permittivity plot saved: {len(E_dielectric)} data points")
    else:
        tqdm.write("Warning: No valid relative permittivity data to plot")
    
    # ==================================================================
    # 【P-Eヒステリシスループのプロット（2回目のループ結果）- ハイライト版】
    # ==================================================================
    tqdm.write("=" * 60)
    tqdm.write("Plotting P-E hysteresis loop (2nd loop results) - Highlighted version")
    tqdm.write("=" * 60)
    
    if len(E_hysteresis) > 0 and len(P_hysteresis) > 0:
        # ハイライト付きP-Eヒステリシスループのプロット
        figsize = (10, 7) if not SPEED_MODE else (8, 6)
        dpi = 300 if not SPEED_MODE else 100
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        markersize = 4 if not SPEED_MODE else 3
        linewidth = 2.5 if not SPEED_MODE else 2.0
        
        # ヒステリシスループ全体を描画
        if not SIMPLE_PLOT_MODE:
            ax.plot(E_hysteresis, P_hysteresis, 'o-', markersize=markersize,
                    linewidth=linewidth, label="2nd: P(E)", 
                    color='blue', antialiased=True)
        else:
            ax.plot(E_hysteresis, P_hysteresis, 'o-', markersize=markersize,
                    linewidth=linewidth, color='blue', antialiased=True)
        
        # DCバイアス範囲（E=0~最大値のpositiveの箇所）を赤線で強調
        E_hyst_arr = np.array(E_hysteresis)
        P_hyst_arr = np.array(P_hysteresis)
        
        # DCバイアス範囲内のデータポイントを抽出（E=0~E_max、positiveの分極）
        dc_bias_mask = (E_hyst_arr >= 0) & (E_hyst_arr <= E_max) & (P_hyst_arr > 0)
        
        if np.any(dc_bias_mask):
            E_dc_bias = E_hyst_arr[dc_bias_mask]
            P_dc_bias = P_hyst_arr[dc_bias_mask]
            
            # 赤線で強調表示（太めの線）
            highlight_linewidth = 3.5 if not SPEED_MODE else 3.0
            ax.plot(E_dc_bias, P_dc_bias, '-', linewidth=highlight_linewidth,
                    color='red', label=f"DC Bias Range (E=0~{E_max:.0f}, P>0)", antialiased=True, zorder=10)
            
            # マーカーも赤で強調
            ax.plot(E_dc_bias, P_dc_bias, 'o', markersize=markersize + 2,
                    color='red', zorder=11)
        
        if not SIMPLE_PLOT_MODE:
            ax.set_title("P vs Electric Field (2nd cycle)")
            ax.legend()
            ax.text(1.05, 0.95, param_text_for_plots, transform=ax.transAxes,
                    fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
        
        # 軸ラベルをシンプルに表示（2nd_P-R.pyと同じスタイル）
        ax.set_xlabel("E")
        ax.set_ylabel("P")
        ax.set_ylim(P_E_graph_y)
        ax.grid()
        
        save_dpi = 300 if not SPEED_MODE else 100
        plt.savefig(os.path.join(GRAPH_SAVE_DIR, "P_E_hysteresis_loop_highlighted.png"), 
                   bbox_inches='tight', dpi=save_dpi)
        plt.close()
        
        tqdm.write(f"Highlighted P-E hysteresis loop plot saved: {len(E_hysteresis)} data points")
        if np.any(dc_bias_mask):
            tqdm.write(f"DC bias range highlighted: {np.sum(dc_bias_mask)} points in range [0, {E_max:.0f}] with positive polarization")
    else:
        tqdm.write("Warning: No valid P-E hysteresis data to plot")

    
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

