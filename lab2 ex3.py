import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# 参数定义
alpha = 2  # coherent state parameter
omega = 50  # oscillator frequency
gamma = 1.5  # damping rate
N_th = 0  # vacuum environment
t = np.linspace(0, 2*np.pi/omega, 1000)  

# 定义算符
a = destroy(50)  # 定义50个基态的算符
H = omega * (a.dag() * a + 0.5)
C1 = np.sqrt(gamma * (N_th + 1)) * a
C2 = np.sqrt(gamma * N_th) * a.dag()

# 定义奇相态
odd_cat_state = (coherent(50, 1j*alpha) - coherent(50, -1j*alpha)).unit()

# 保真度和维格纳负性函数
fidelities = []
wigner_negativities = []
r_values = np.linspace(0, 2, 50)  # 定义不同的r值范围

# 选择方向，根据前一个练习中的结果，我们假设P方向 (phi = π/2)
phi = np.pi

for r in r_values:
    # 压缩态
    squeezed_cat = squeeze(50, r * np.exp(1j * phi)) * odd_cat_state

    # 演化
    result = mesolve(H, squeezed_cat, t, c_ops=[C1, C2], e_ops=[])

    # 逆压缩操作
    inverse_squeezing = squeeze(50, -r * np.exp(1j * phi))
    final_state = inverse_squeezing * result.states[-1]

    # 计算保真度
    fidelity_val = fidelity(odd_cat_state, final_state)
    fidelities.append(fidelity_val)

    # 计算维格纳函数并找出最小值
    x = np.linspace(-5, 5, 200)
    W = wigner(final_state, x, x)
    wigner_negativity = W.min()
    wigner_negativities.append(wigner_negativity)

# 绘制保真度和维格纳负性作为r的函数
plt.figure(figsize=(12, 5))

# 保真度
plt.subplot(1, 2, 1)
plt.plot(r_values, fidelities, label='Fidelity')
plt.xlabel('Squeezing parameter r')
plt.ylabel('Fidelity')
plt.title('Fidelity vs Squeezing parameter')
plt.grid(True)
plt.legend()

# 维格纳负性
plt.subplot(1, 2, 2)
plt.plot(r_values, wigner_negativities, label='Wigner Negativity', color='orange')
plt.xlabel('Squeezing parameter r')
plt.ylabel('Wigner Negativity')
plt.title('Wigner Negativity vs Squeezing parameter')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 找到最佳r值
optimal_r_fidelity = r_values[np.argmax(fidelities)]
optimal_r_wigner_negativity = r_values[np.argmin(wigner_negativities)]

print(f"Optimal r for Fidelity: {optimal_r_fidelity}")
print(f"Optimal r for Wigner Negativity: {optimal_r_wigner_negativity}")

