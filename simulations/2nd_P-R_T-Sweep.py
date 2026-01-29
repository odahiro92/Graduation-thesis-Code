#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
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
SPEED_MODE = False

# True: シンプル描画モード（タイトル、ボックス、凡例を非表示、プロットと軸目盛り・軸タイトルのみ表示）
SIMPLE_PLOT_MODE = False
 
# ----------- -------------------------------------------------------
# 2. 物理パラメータ（自由エネルギー計算の係数）
# ------------------------------------------------------------------
fixed_params = { 
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
    'gamma_82': 0.0,            # γ_82*P^8*R^2項の係数（新規追加）
    'gamma_64': -0.000000,      # γ_64*P^6*R^4項の係数
    'gamma_84': 0.0000000005,   # γ_84*P^8*R^4項の係数
    'gamma_86': 0.00000000000,  # γ_86*P^8*R^6項の係数
}

# ------------------------------------------------------------------
# 3. プロット生成制御フラグ
# ------------------------------------------------------------------
# このプログラムは各温度でのP-E、R-Eヒステリシスループの静止画像を生成し、
# それらを温度順にまとめてGIFアニメーションとして出力します
SHOW_R_COEFFICIENTS_BOX = False  # R^2, R^4, R^6, Sum, Dのボックスを表示するか（ヒステリシスループでは使用しない）
# ------------------------------------------------------------------
# 4. 温度範囲設定
# ------------------------------------------------------------------
t_min = -50.0  # 温度範囲の最小値
t_max = 50.0   # 温度範囲の最大値
t_divisions = 30 if SPEED_MODE else 200  # 温度範囲を何分割するか（スピードモード: 5, 品質モード: 10）

# ------------------------------------------------------------------
# 4-2. 電場範囲設定
# ------------------------------------------------------------------
E_range_abs = 400.0  # 電場範囲の絶対値（最大値は正、最小値は負になる）
E_divisions = 25 if SPEED_MODE else 60  # 電場範囲を何分割するか（スピードモード: 25, 品質モード: 60）

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
ENERGY_CUTOFF = 2000.0  # エネルギーカットオフ値

# ------------------------------------------------------------------
# 7. 保存先ディレクトリ設定
# ------------------------------------------------------------------
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "Outputs_t")
# 描画モードとスピードモードでフォルダ名を変更
# シンプル描画モードの場合は、temp_t_speed/temp_t_qualityの中にsimple_modeディレクトリを作成
# シンプル描画モードがTrueの時はシンプル描画モードの出力のみ、Falseの時は通常モードのみの出力
if SPEED_MODE:
    base_dir = os.path.join(OUTPUTS_DIR, "temp_t_speed")
else:
    base_dir = os.path.join(OUTPUTS_DIR, "temp_t_quality")

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

# 後方互換性のための設定（P_range, R_rangeから自動設定）
P_grid = P_range
R_grid = R_range
contour_graph_x = P_range
contour_graph_y = R_range
P_E_graph_y = P_range
R_E_graph_y = R_range

# 温度範囲（単純な線形分割）
t_range = np.linspace(t_min, t_max, t_divisions + 1)

# 電場範囲の最大値・最小値
E_max = E_range_abs
E_min = -E_range_abs

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
def calc_r_coefficients(P_at_stable, t, params):
    """R^2, R^4, R^6の係数値を計算する関数（重複計算を避けるため）"""
    beta_1_eff = params['beta_1'] * (t - params['t_{cR}'])
    P2 = P_at_stable ** 2
    P4 = P2 ** 2
    P6 = P4 * P2
    P8 = P4 ** 2
    
    R2_val = (
        0.5 * beta_1_eff
        + params['gamma_22'] * P2
        + params['gamma_42'] * P4
        + params['gamma_82'] * P8
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
def calc_free_energy(P, R, t, E, params):
    """自由エネルギーを計算（温度tと電場Eの両方を受け取る）"""
    alpha_1_t = params['alpha_1'] * (t - params['t_{cP}'])
    beta_1_eff = params['beta_1'] * (t - params['t_{cR}'])
    # べき乗計算を最適化（P**2, P**4, P**6, P**8を一度計算して再利用）
    P2 = P**2
    P4 = P2**2
    P6 = P4 * P2
    P8 = P4**2
    R2 = R**2
    R4 = R2**2
    R6 = R4 * R2
    return (
        0.5 * alpha_1_t * P2
        + 0.25 * params['alpha_2'] * P4
        + (1.0/6.0) * params['alpha_3'] * P6
        + 0.5 * beta_1_eff * R2
        + 0.25 * params['beta_2'] * R4
        + (1.0/6.0) * params['beta_3'] * R6
        + params['gamma_22'] * P2 * R2
        + params['gamma_42'] * P4 * R2
        + params['gamma_82'] * P8 * R2
        + params['gamma_64'] * P6 * R4
        + params['gamma_84'] * P8 * R4
        + params['gamma_86'] * P8 * R6
        - P * E  # 電場Eを直接使用
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


def plot_hysteresis_P_E_static(E_array, P_stable_array, t, n, subdir, params, param_text="", simple_mode=None):
    """各温度での完全なP-Eヒステリシスループを1枚の静止画像として描画"""
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ヒステリシスループ全体を描画
    ax.plot(E_array, P_stable_array, 'o-', markersize=2, linewidth=2.5, 
            color='blue', alpha=0.9, label="P(E) loop")
    
    if not simple_mode:
        ax.set_title(f"P vs Electric Field (Hysteresis Loop)")
        ax.legend(loc='best', fontsize=8)
        
        ax.text(1.05, 0.95, param_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
    
    # 軸ラベルを大きく太字で表示
    label_fontsize = 22 if not SPEED_MODE else 20
    ax.set_xlabel("E", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel("P", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylim(P_E_graph_y)
    ax.grid(True, alpha=0.3)
    
    # 温度情報を別ボックスで表示（凡例風のスタイル、マイナスの時は太字）
    fontweight = 'bold' if t < 0 else 'normal'
    ax.text(0.02, 0.98, f"t = {t:.1f}", transform=ax.transAxes,
            fontsize=16, fontweight=fontweight, verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7))
    
    # ディレクトリはメインループで既に作成済み
    filepath = os.path.join(subdir, f"hysteresis_P_E_t_{n:04d}.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    
    # ファイルが完全に書き込まれるまで待つ
    max_wait = 5.0  # 最大5秒待つ
    wait_interval = 0.1
    waited = 0.0
    while waited < max_wait:
        if os.path.exists(filepath):
            try:
                file_size = os.path.getsize(filepath)
                if file_size > 0:
                    break
            except OSError:
                pass
        time.sleep(wait_interval)
        waited += wait_interval

def plot_hysteresis_R_E_static(E_array, R_stable_array, t, n, subdir, params, param_text="", simple_mode=None):
    """各温度での完全なR-Eヒステリシスループを1枚の静止画像として描画"""
    if simple_mode is None:
        simple_mode = SIMPLE_PLOT_MODE
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ヒステリシスループ全体を描画
    ax.plot(E_array, R_stable_array, 'o-', markersize=2, linewidth=2.5, 
            color='green', alpha=0.9, label="R(E) loop")
    
    if not simple_mode:
        ax.set_title(f"R vs Electric Field (Hysteresis Loop)")
        ax.legend(loc='best', fontsize=8)
        
        ax.text(1.05, 0.95, param_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
    
    # 軸ラベルを大きく太字で表示
    label_fontsize = 22 if not SPEED_MODE else 20
    ax.set_xlabel("E", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel("R", fontsize=label_fontsize, fontweight='bold')
    ax.set_ylim(R_E_graph_y)
    ax.grid(True, alpha=0.3)
    
    # 温度情報を別ボックスで表示（凡例風のスタイル、マイナスの時は太字）
    fontweight = 'bold' if t < 0 else 'normal'
    ax.text(0.02, 0.98, f"t = {t:.1f}", transform=ax.transAxes,
            fontsize=16, fontweight=fontweight, verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7))
    
    # ディレクトリはメインループで既に作成済み
    filepath = os.path.join(subdir, f"hysteresis_R_E_t_{n:04d}.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    
    # ファイルが完全に書き込まれるまで待つ
    max_wait = 5.0  # 最大5秒待つ
    wait_interval = 0.1
    waited = 0.0
    while waited < max_wait:
        if os.path.exists(filepath):
            try:
                file_size = os.path.getsize(filepath)
                if file_size > 0:
                    break
            except OSError:
                pass
        time.sleep(wait_interval)
        waited += wait_interval

# 以下の関数は使用されていないため削除（P-E、R-Eヒステリシスループのみ生成）
# get_perovskite_structure, apply_polarization_and_rotation, plot_3d_unitcell_oblique, plot_3d_unitcell_topview

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
    # fpsからduration（秒）に変換して使用（GIFの仕様に合わせる）
    fps = 8  # 両モード共通でfps=8（少し遅めの再生速度）
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
        # writerの作成パラメータ（durationを使用して確実にfpsを制御）
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

# create_summary_gif関数は使用されていないため削除（P-E、R-Eヒステリシスループのみ生成）

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
    print(f"Temperature range: {t_min} to {t_max} ({len(t_range)} points)")
    print(f"Electric field range: {E_min} to {E_max} ({len(E_range)} points per temperature)")
    
    # ターミナル出力のバッファリングを無効化（プログレスバーが確実に表示されるように）
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except:
            pass
    
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
    
    # 電場範囲の1周目と2周目の境界
    n_E_1st = len(E_range_1st)  # 1周目の電場数
    n_E_2nd = len(E_range_2nd)  # 2周目の電場数
    
    # メインループ：外側で温度をループ、内側で電場をループ
    tqdm.write("Running main simulation loop (t × E)...")
    total_steps = len(t_range) * len(E_range)
    main_pbar = tqdm(total=total_steps, desc="Main simulation", 
                     mininterval=0.1, maxinterval=1.0, leave=True)
    
    # 各温度での結果を保存
    all_results = {}  # {t: {'E_array': [...], 'P_array': [...], 'R_array': [...]}}
    
    for t_idx, t in enumerate(t_range):
        # この温度での電場ループ用の変数
        E_array = []
        P_stable_array = []
        R_stable_array = []
        
        current_p = init_P_value
        current_r = init_R_value
        
        # この温度でのPとRの最大値を計算（2周目のデータのみ）
        P_max_temp = 0.0
        R_max_temp = 0.0
        
        # 電場ループ（ヒステリシスループ）
        for E_idx, E in enumerate(E_range):
            # 自由エネルギーを計算
            free_energy = calc_free_energy(P_mesh, R_mesh, t, E, fixed_params)
            free_energy_cut = np.where(free_energy > ENERGY_CUTOFF, ENERGY_CUTOFF, free_energy)
            
            # 安定点を探索
            stable_p, stable_r = find_stable_point(P_mesh, R_mesh, free_energy_cut, current_p, current_r)
            
            # NaN/Infのガード処理
            if not np.isfinite(stable_p) or not np.isfinite(stable_r):
                if len(P_stable_array) > 0:
                    stable_p = P_stable_array[-1]
                    stable_r = R_stable_array[-1]
                else:
                    stable_p = 0.0
                    stable_r = 0.0
            
            E_array.append(E)
            P_stable_array.append(stable_p)
            R_stable_array.append(stable_r)
            
            # 2周目のデータのみで最大値を更新
            if E_idx >= n_E_1st:
                P_max_temp = max(P_max_temp, abs(stable_p))
                R_max_temp = max(R_max_temp, abs(stable_r))
            
            current_p = stable_p
            current_r = stable_r
            
            main_pbar.update(1)
            
            # メモリリーク対策
            plt.close('all')
            if E_idx % 10 == 0:  # 10回に1回だけガベージコレクション
                gc.collect()
        
        # この温度での結果を保存
        all_results[t] = {
            'E_array': E_array,
            'P_array': P_stable_array,
            'R_array': R_stable_array
        }
        
        # この温度でのP-E、R-Eヒステリシスループを1枚の静止画像として描画（2周目のデータのみ）
        E_array_2nd = E_array[n_E_1st:]
        P_array_2nd = P_stable_array[n_E_1st:]
        R_array_2nd = R_stable_array[n_E_1st:]
        
        # 各温度での静止画像を生成
        tqdm.write(f"Generating static hysteresis plots for t={t:.1f}...")
        plot_hysteresis_P_E_static(E_array_2nd, P_array_2nd, t, t_idx, GRAPH_SAVE_DIR, fixed_params, 
                                   param_text=param_text_for_plots, simple_mode=SIMPLE_PLOT_MODE)
        plot_hysteresis_R_E_static(E_array_2nd, R_array_2nd, t, t_idx, GRAPH_SAVE_DIR, fixed_params, 
                                   param_text=param_text_for_plots, simple_mode=SIMPLE_PLOT_MODE)
    
    # メインシミュレーションのプログレスバーを閉じる
    main_pbar.close()
    
    # 全ての温度での静止画像を収集してGIFを作成
    tqdm.write("Creating temperature sweep GIF animations...")
    create_animation(GRAPH_SAVE_DIR, "hysteresis_P_E_t_sweep.gif", "hysteresis_P_E_t_", start_index=0, pbar=None)
    create_animation(GRAPH_SAVE_DIR, "hysteresis_R_E_t_sweep.gif", "hysteresis_R_E_t_", start_index=0, pbar=None)
    
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