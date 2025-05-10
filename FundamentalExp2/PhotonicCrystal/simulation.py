import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import cmath
import math
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['LXGW WenKai'] # 解决中文不显示问题
plt.rcParams['axes.unicode_minus'] = False

# 定义 alpha, beta, gamma 的计算公式
def alpha_calc(f):
    return 1.810e-6 * math.sqrt(f)
def beta_calc(epsilon_r, mu_r, f):
    return 2 * math.pi * f * math.sqrt(epsilon_r * 8.854187817e-12 * 4e-7 * math.pi * mu_r)
def gamma_calc(alpha, beta):
    return complex(alpha, beta)

# 定义 A 矩阵元的计算公式
class A(np.ndarray):
    def __new__(cls, a, b, c, d):
        data = np.array([
            [complex(a), complex(b)],
            [complex(c), complex(d)]
        ], dtype = complex)
        return data.view(cls)

    # 添加属性访问
    @property
    def a(self): return self[0, 0]
    @property
    def b(self): return self[0, 1]
    @property
    def c(self): return self[1, 0]
    @property
    def d(self): return self[1, 1]

    def __repr__(self):
        return f"A(\n{np.array_str(self)}\n)"

# 定义矩阵元计算公式
def a_calc(gamma, l):
    return complex(cmath.cosh(gamma*l))
def b_calc(Z, gamma, l):
    return complex(cmath.sinh(gamma*l)*Z)
def c_calc(Z, gamma, l):
    return complex(cmath.sinh(gamma*l)/Z)
def d_calc(gamma, l):
    return complex(cmath.cosh(gamma*l))

# 定义 v_L 的计算公式
def v_L_calc(a, b, c, d, Z):
    return 1/(a + b/Z + c*Z + d)

# 主程序
def main():
    try:
        epsilon_r = float(input("输入相对介电常数: "))
        mu_r = float(input("输入相对磁导率: "))
        z_0 = float(input("输入阻抗："))
        l = float(input("输入一段电缆的长度："))
        mixing = int(input("是否有掺杂？(0) 无 (1) 有  "))
        if mixing not in [0, 1]:
            raise ValueError("输入必须为 0 或 1")
        
    except ValueError:
        print("输入无效，请输入有效的数字！")
        return

    # 定义频率范围 f 从 1 到 60
    frequencies = np.arange(1e6, 61e6, 1e6) 
    size = 60
    v_L_values = np.zeros(size, dtype = complex)

    # 计算每个频率对应的 v_L 值
    for f in frequencies:
        # 计算 alpha, beta, gamma
        beta = beta_calc(epsilon_r, mu_r, f)
        alpha = alpha_calc(f)
        gamma = gamma_calc(alpha, beta)

        # 计算矩阵元
        a = a_calc(gamma, l)
        b = b_calc(z_0, gamma, l)
        c = c_calc(z_0, gamma, l)
        d = d_calc(gamma, l)

        # 计算 A_T
        A_1 = A(a, b, c, d)
        A_2 = A(a, 2*b, c/2, d)
        A_T = A_2 @ A_1 @ A_2 @ A_1 @ A_2 @ A_1 @ A_2
        A_T_prime = A_2 @ A_1 @ A_2 @ A_1 @ A_1 @ A_2 @ A_1 @ A_2
        
        if mixing == 0:
            # 写出矩阵元
            a_T = A_T.a
            b_T = A_T.b
            c_T = A_T.c
            d_T = A_T.d
            
            # 计算 v_L
            v_L_values[int(f/(1e6)-1)] = v_L_calc(a_T, b_T, c_T, d_T, z_0)
        
        if mixing == 1:
            # 写出矩阵元
            a_T = A_T_prime.a
            b_T = A_T_prime.b
            c_T = A_T_prime.c
            d_T = A_T_prime.d
            
            # 计算 v_L
            v_L_values[int(f/(1e6)-1)] = v_L_calc(a_T, b_T, c_T, d_T, z_0)

    # 转换为 NumPy 数组
    v_L_values = np.array(v_L_values)
    
    # 计算电压幅值并排序（确保x单调）
    eta = 4*(np.abs(v_L_values))**2
    frequencies_mhz = frequencies / 1e6
    sort_idx = np.argsort(frequencies_mhz)
    x_ordered = frequencies_mhz[sort_idx]
    y_ordered = eta[sort_idx]
    
    # 生成密集插值点
    x_new = np.linspace(x_ordered.min(), x_ordered.max(), 500)
    spl = make_interp_spline(x_ordered, y_ordered, k=3)  # 三次样条
    y_smooth = spl(x_new)

    # 从 v_L_values 中提取相位差
    phase = np.angle(v_L_values)
    phase_correct = np.unwrap(phase)

    # 计算等效折射率 n = (φ * c) / (2π * f * L)
    frequencies_hz = frequencies
    n = (abs(phase_correct) * 3e8) / (2 * np.pi * frequencies_hz * (7 + mixing) * l)

    # 计算群速度 v_g = c / (n + ω * dn/dω)
    omega = 2 * np.pi * frequencies_hz
    dn = np.diff(n)
    domega = np.diff(omega)
    v_g = 3e8 / (n[:-1] + omega[:-1] * (dn / domega))

    # 对齐数据索引（群速度比频率少一个点）
    f_vg = frequencies_mhz[:-1][sort_idx[:-1]]
    v_g_ordered = v_g[sort_idx[:-1]]
    n_ordered = n[sort_idx]

    # ----------------------------- 样条插值处理 -----------------------------
    # 等效折射率样条插值
    spl_n = make_interp_spline(x_ordered, n_ordered, k=3)
    x_new_n = np.linspace(x_ordered.min(), x_ordered.max(), 500)
    y_smooth_n = spl_n(x_new_n)

    # 群速度样条插值
    spl_vg = make_interp_spline(f_vg, v_g_ordered, k=3)
    x_new_vg = np.linspace(f_vg.min(), f_vg.max(), 500)
    y_smooth_vg = spl_vg(x_new_vg)

    # ----------------------------- 绘图 -----------------------------
    fig, ax1 = plt.subplots(figsize=(12,7))

    # 传输效率 (左轴) - 保留原有样条曲线
    ax1.scatter(x_ordered, y_ordered, color="royalblue", edgecolor="navy", label=r"$\eta$ (原始)")
    ax1.plot(x_new, y_smooth, color="royalblue", linewidth=2, label=r"$\eta$ (样条拟合)")

    # 等效折射率 (右轴) - 新增样条曲线
    ax2 = ax1.twinx()
    ax2.scatter(x_ordered, n_ordered, color="crimson", marker="s", edgecolor="darkred", label=r"$n$ (原始)")
    ax2.plot(x_new_n, y_smooth_n, color="crimson", linestyle="-", linewidth=2, label=r"$n$ (样条拟合)")

    # 群速度 (次右轴) - 新增样条曲线
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15)) 
    ax3.scatter(f_vg, v_g_ordered/1e8, color="forestgreen", marker="^", edgecolor="darkgreen", label=r"$v_g$ (原始)")
    ax3.plot(x_new_vg, y_smooth_vg/1e8, color="forestgreen", linestyle="-", linewidth=2, label=r"$v_g$ (样条拟合)")

    # 坐标轴标签和样式
    ax1.set_xlabel("频率 (MHz)", fontsize=12)
    ax1.set_ylabel(r"传输效率 $\eta$", color="royalblue", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="royalblue")
    ax2.set_ylabel(r"等效折射率 $n$", color="crimson", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="crimson")
    ax3.set_ylabel(r"群速度 $v_g$ ($\times 10^8$ m/s)", color="forestgreen", fontsize=12)
    ax3.tick_params(axis='y', labelcolor="forestgreen")

    # 图表修饰
    plt.title("同轴光子晶体综合传输特性 (全样条拟合)", fontsize=14, pad=20)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1+lines2+lines3, labels1+labels2+labels3, 
            loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, 61)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    
# 运行主程序
if __name__ == "__main__":
    main()