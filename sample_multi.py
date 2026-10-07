import numpy as np
from pytopo3d.core.optimizer import top3d_multi
from pytopo3d.utils.export import voxel_to_stl  # 公式ユーティリティのインポート
from pytopo3d.utils.results_manager import ResultsManager
from pytopo3d.visualization.visualizer import (
    visualize_final_result,
    visualize_initial_setup,
)

# --------------------------------------------------
# 1. パラメータとグリッド解像度
# --------------------------------------------------
nelx = 60  # x方向（垂直壁から離れる奥行き）
nely = 30  # y方向（幅）
nelz = 40  # z方向（高さ）

volfrac = 0.25   # 体積比率（25%）
penal = 3.0      # SIMPペナルティ係数
rmin = 2.5       # フィルタ半径
disp_thres = 0.5 # 固体判定しきい値

# --------------------------------------------------
# 2. 荷重条件 (平面2: 先端の水平面に下向き荷重)
#    配列形状: (nely, nelx, nelz, 3) -> [Fx, Fy, Fz]
# --------------------------------------------------
force_fields = np.zeros((2, nely, nelx, nelz, 3), dtype=np.float64)
x_start = int(nelx * 0.75)  # 壁から離れた先端側
#force_field[5:25, x_start:nelx, 0:2, 2] = -1.0  # -z 方向荷重
force_fields[0, 5:25, x_start:nelx, 0:2, 2] = -0.4  # -z 方向荷重
force_fields[1, 5:25, x_start:nelx, 0:2, 0] = -0.4  # -x 方向荷重

# --------------------------------------------------
# 3. 障害物マスク (干渉領域のくり抜き)
#    配列形状: (nely, nelx, nelz) -> True は材料配置禁止
# --------------------------------------------------
obstacle_mask = np.zeros((nely, nelx, nelz), dtype=bool)
obstacle_mask[:, 30:nelx, 25:nelz] = True  # L字空間を誘導

# 可視化用の荷重・拘束配列を作成
experiment_name = "multi_force"
results_mgr = ResultsManager(experiment_name=experiment_name)
loads_array = np.any(force_fields != 0, axis=(0, 4)).astype(float)
constraints_array = np.zeros((nely, nelx, nelz), dtype=float)
constraints_array[5:25, 0, 10:30] = 1.0  # x=0 面の指定領域を支持

setup_viz_path = visualize_initial_setup(
    nelx=nelx,
    nely=nely,
    nelz=nelz,
    loads_array=loads_array,
    constraints_array=constraints_array,
    experiment_name=experiment_name,
    results_mgr=results_mgr,
    combined_obstacle_mask=obstacle_mask,
)
print(f"初期条件の可視化を出力しました: {setup_viz_path}")

# --------------------------------------------------
# 4. 最適化計算の実行
# --------------------------------------------------
print("トポロジー最適化を実行中...")
density_result = top3d_multi(
    nelx=nelx,
    nely=nely,
    nelz=nelz,
    volfrac=volfrac,
    penal=penal,
    rmin=rmin,
    disp_thres=disp_thres,
    obstacle_mask=obstacle_mask,
    force_fields=force_fields,
    support_mask=constraints_array.astype(bool),
    maxloop=200,
    use_gpu=True,
)

result_viz_path = visualize_final_result(
    nelx=nelx,
    nely=nely,
    nelz=nelz,
    experiment_name=experiment_name,
    disp_thres=disp_thres,
    results_mgr=results_mgr,
    xPhys=density_result,
    combined_obstacle_mask=obstacle_mask,
    loads_array=loads_array,
    constraints_array=constraints_array,
)
print(f"最適化結果の可視化を出力しました: {result_viz_path}")

# --------------------------------------------------
# 5. voxel_to_stl を用いた STL エクスポート
# --------------------------------------------------
output_stl = f"{experiment_name}.stl"

# 内部で自動的に軸変換 (nely, nelx, nelz) -> (x, y, z) が行われます
voxel_to_stl(
    density_result,
    output_file=output_stl,
    level=disp_thres,
    smooth_mesh=True,
    smooth_iterations=10
)

print(f"STLファイルを出力しました: {output_stl}")
