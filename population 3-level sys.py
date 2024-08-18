import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# 定义系统参数
omega_eg = 1.0  # 两个基态到激发态的跃迁频率
omega_ge = 0.9  # 激发态到基态的跃迁频率
delta = 0.2     # 驱动场的失谐参数

# 定义系统的哈密顿量
H0 = omega_eg * basis(3, 0) * basis(3, 0).dag() + omega_eg * basis(3, 1) * basis(3, 1).dag() + omega_ge * basis(3, 2) * basis(3, 2).dag()
H1 = (basis(3, 0) * basis(3, 2).dag() + basis(3, 1) * basis(3, 2).dag() + basis(3, 2) * basis(3, 0).dag() + basis(3, 2) * basis(3, 1).dag())
H_drive = 0.5 * (H1 + H1.dag())  # 驱动哈密顿量

# 定义驱动脉冲
tlist = np.linspace(0, 10, 1000)
args = {'omega_eg': omega_eg, 'omega_ge': omega_ge, 'delta': delta}

# 演化系统的时间演化
result = mesolve(H0 + delta * H_drive, basis(3, 0), tlist, [], [basis(3, 0) * basis(3, 0).dag(), basis(3, 1) * basis(3, 1).dag(), basis(3, 2) * basis(3, 2).dag()], args=args)

# 绘制结果
plt.plot(tlist, result.expect[0], label=r'$\rho_{g1}$')
plt.plot(tlist, result.expect[1], label=r'$\rho_{g2}$')
plt.plot(tlist, result.expect[2], label=r'$\rho_{e}$')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Population Dynamics of a Three-Level System')
plt.legend()
plt.show()

