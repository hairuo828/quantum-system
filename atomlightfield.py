from qutip import jmat,mesolve,fock,spin_state,expect,tensor,qeye,destroy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 15,
    "mathtext.fontset":'cm',
    "font.serif": ['Arial'],
}

rcParams.update(config)

j = 1/2
Jp=jmat(j,'+')#原子的升算符
J_=jmat(j,'-')#原子的降算符
Jx=jmat(j,'x')#原子的Jx算符
Jy=jmat(j,'y')#原子的Jy算符,这里的j是虚数单位
Jz=jmat(j,'z')#原子的Jz算符

a=destroy(50)#光场的湮灭算符,50是矩阵的维度。理论上维度应该是无限大,但实际计算只能给定足够大的有限维度
a_plus=a.dag()#光场的产生算符
eye=qeye(50)#单位阵
psi0 = tensor(fock(50,0),spin_state(j,0))#设置系统的初态为直积态 |0>×|↑>

H=tensor(a,Jp)+tensor(a_plus,J_)#系统的哈密顿量
tlist=np.linspace(0,10,1000)#时间列表
result=mesolve(H,psi0,tlist)#态随时间的演化

fig=plt.figure(figsize=(8,6))

(ax1,ax2),(ax3,ax4) = fig.subplots(2,2)
ax1.plot(tlist,expect(tensor(eye,Jx),result.states),color='blue');ax1.set_ylabel(r"$\langle \hat{J}_x \rangle$")#Jx的平均值随时间变化图
ax2.plot(tlist,expect(tensor(eye,Jz),result.states),color='blue');ax2.set_ylabel(r"$\langle \hat{J}_z \rangle$")#Jz的平均值随时间变化图
ax3.plot(tlist,expect(tensor(a_plus*a,qeye(2)),result.states),color='blue');ax3.set_ylabel(r"$\langle \hat{a}^{\dag} a \rangle$")#光子数的平均值随时间变化图
ax4.plot(tlist,expect(tensor(eye,Jx**2+Jy**2+Jz**2),result.states),color='blue');ax4.set_ylabel(r"$\langle \hat{J}^2 \rangle$")#J平方的平均值随时间变化图

for ax in fig.axes:
    ax.tick_params(direction="in",axis='both',labelsize=15,length=4,width=1.2)
    ax.set_xlim(0,tlist[-1])
    [spine.set_linewidth(1.2) for spine in ax.spines.values()]
    ax.set_xlabel(r"$t$",size=20)
ax1.set_ylim(-1.1,1.1)
ax4.set_ylim(0,2.2)
fig.tight_layout()
#fig.subplots_adjust(top=None,bottom=None,left=None,right=None,wspace=0.4,hspace=0.4)
fig.show()

