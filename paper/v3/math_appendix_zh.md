# 附录D 指令保持视觉对比解码：定义、命题与完整证明

本附录从共享生成前缀下的条件分布出发，推导候选准入、参考指令交互与IP-VCD的保持性质，并给出三条件线性族的系数刻画、固定支持上的变分表达、候选组概率及完整回复概率的证明。VCD的视觉对比与候选约束采用Leng等人的定义[1]；三条件构造及其输入职责沿用本文方法。下述等式均明确保留对应的候选支持与归一化。

## D.1 条件分布、候选支持与归一化

### D.1.1 四个条件与共享前缀

设冻结的视觉语言模型为$p_\theta$，有限词表为$\mathcal V$且$|\mathcal V|\geq2$，正常图像为$I$，参考图像为$\widetilde I$，当前已生成的词元前缀为$y_{<t}$。在一次条件比较中，$\widetilde I$及其扰动随机量保持固定。令$x_0$包含任务问题与简洁回答要求，$x_1$在$x_0$上增加指定的弃权条件句。四个条件的下一词元对数概率定义为

$$
\begin{aligned}
c_t(v)&=\log p_\theta(v\mid I,x_0,y_{<t}),&
r_t(v)&=\log p_\theta(v\mid\widetilde I,x_0,y_{<t}),\\
g_t(v)&=\log p_\theta(v\mid I,x_1,y_{<t}),&
h_t(v)&=\log p_\theta(v\mid\widetilde I,x_1,y_{<t}).
\end{aligned}
\tag{D.1}
$$

四路使用相同模型、词表与实际回复前缀，各自保留与其输入条件对应的缓存。设各分支在$\mathcal V$上的未截断logit有限，因此其softmax概率为正。局部推导固定$t$并省略时间下标，记

$$
p_c=e^c,\quad p_r=e^r,\quad p_g=e^g,\quad p_h=e^h,
\qquad
\delta_c=g-c,\quad\delta_r=h-r,\quad d=c-r.
\tag{D.2}
$$

$\delta_c$与$\delta_r$逐词元记录同一弃权条件句在两种视觉输入上引起的对数概率变化。$d$比较正常图像与参考图像对同一续写的支持；它的两个条件都保留任务问题、简洁回答要求和当前回复前缀。

### D.1.2 候选约束与解码分布

对对数概率向量$\ell$和$0<\beta\leq1$，定义

$$
\mathcal S_\ell
=\{v\in\mathcal V:\ell(v)\geq\max_{w\in\mathcal V}\ell(w)+\log\beta\}
=\{v:p_\ell(v)\geq\beta\max_w p_\ell(w)\}.
\tag{D.3}
$$

当$\beta=0$时，约定$\mathcal S_\ell=\mathcal V$。任意最大概率词元均属于$\mathcal S_\ell$，所以该集合非空。对非空集合$S\subseteq\mathcal V$和有限分数$s$，定义受限归一化算子

$$
\mathcal P_S[s](v)
=\frac{\mathbf1\{v\in S\}\exp s(v)}{Z_S(s)},
\qquad Z_S(s)=\sum_{w\in S}\exp s(w).
\tag{D.4}
$$

三个主要解码条件分别为

$$
\begin{aligned}
s_N&=(1+\alpha)c-\alpha r,&p_N&=\mathcal P_{\mathcal S_c}[s_N],\\
s_G&=(1+\alpha)g-\alpha h,&p_G&=\mathcal P_{\mathcal S_g}[s_G],\\
s_{\mathrm{IP}}&=g+\alpha(c-r),&p_{\mathrm{IP}}&=\mathcal P_{\mathcal S_g}[s_{\mathrm{IP}}],
\end{aligned}
\qquad \alpha\geq0.
\tag{D.5}
$$

下标$N$、$G$与$\mathrm{IP}$依次表示原生无弃权引导VCD、带引导VCD与IP-VCD。它们的分数函数定义于整个词表，实际归一化支持由式（D.5）指定。本文贪心生成在每一步选择该支持内分数最高的词元；并列时采用实现登记的固定顺序。后文的概率恒等式讨论式（D.4）所定义的归一化分数分布，词元选择由相应的最大值条件给出。

**引理D.1（归一化常数的消去）。** 对任意$u,v\in S$，有

$$
\log\frac{\mathcal P_S[s](u)}{\mathcal P_S[s](v)}=s(u)-s(v).
\tag{D.6}
$$

**证明。** 两个概率的分母同为$Z_S(s)$，取比值后分母抵消，再取对数即可。进一步，若第$j$个分支的logit为$z_j$，则$\ell_j=z_j-\log\sum_v e^{z_j(v)}$。对任意与词元无关的系数$a_j$，$\sum_j a_jz_j$与$\sum_j a_j\ell_j$相差一个与词元无关的常数。受限softmax、候选间的分数差及argmax均保持相同。候选准入比较$\ell_j(v)-\max_w\ell_j(w)$，同样消去该常数。证毕。

这一性质使原始logit实现与对数概率推导对应起来，也使以下保持性质可以写成实际归一化分布的对数优势比。

## D.2 两阶段作用：候选准入与参考指令交互

### D.2.1 弃权候选的准入条件

对$0<\beta\leq1$，定义候选$u$距准入阈值的差

$$
K_\ell(u)=\ell(u)-\max_w\ell(w)-\log\beta.
\tag{D.7}
$$

$K_\ell(u)\geq0$恰好对应$u\in\mathcal S_\ell$。由于$g=c+\delta_c$，有

$$
K_g(u)-K_c(u)
=\delta_c(u)-\big[\max_w\{c(w)+\delta_c(w)\}-\max_w c(w)\big].
\tag{D.8}
$$

**证明。** 将$g=c+\delta_c$分别代入$K_g$与$K_c$，消去$c(u)$和$\log\beta$即得。等价地，候选准入可以写成所有竞争词元上的条件

$$
u\in\mathcal S_g
\iff
[c(u)-c(v)]+[\delta_c(u)-\delta_c(v)]\geq\log\beta,
\quad\forall v\in\mathcal V.
\tag{D.9}
$$

因为$g(u)\geq\max_vg(v)+\log\beta$等价于$g(u)-g(v)\geq\log\beta$对所有$v$成立。证毕。

式（D.8）表明，指令对弃权候选的增量要与词表最大分数的移动共同比较。候选的相对位置越过阈值后，才进入后续的分数竞争。由式（D.4），支持之外候选的概率为零；支持内候选的相对概率由$s$确定。IP-VCD的准入集合由$g$给出，视觉参考$r$参与其上的分数调整。

### D.2.2 带引导VCD的完整分解

**命题D.1（参考指令交互）。** 四路条件满足

$$
\boxed{
s_G=s_N+\delta_c+\alpha(\delta_c-\delta_r),\qquad
s_{\mathrm{IP}}=s_N+\delta_c.
}
\tag{D.10}
$$

**证明。** 使用$g=c+\delta_c$和$h=r+\delta_r$，逐项展开：

$$
\begin{aligned}
s_G
&=(1+\alpha)(c+\delta_c)-\alpha(r+\delta_r)\\
&=(1+\alpha)c-\alpha r+(1+\alpha)\delta_c-\alpha\delta_r\\
&=s_N+\delta_c+\alpha(\delta_c-\delta_r),\\
s_{\mathrm{IP}}
&=c+\delta_c+\alpha(c-r)
=s_N+\delta_c.
\end{aligned}
\tag{D.11}
$$

两式相减得到

$$
s_{\mathrm{IP}}-s_G=\alpha(\delta_r-\delta_c).
\tag{D.12}
$$

证毕。该分解将原生视觉对比、正常图像上的一次指令增量以及额外的参考交互分别列出。相同指令在不同视觉条件上的作用由两条实际条件分布给定。

### D.2.3 从分数变化到候选翻转

固定一个弃权续写候选$u$和一个答案续写候选$v$，统一以“弃权减答案”为边际方向：

$$
m_\ell=\ell(u)-\ell(v),\qquad
\Delta_c=\delta_c(u)-\delta_c(v),\qquad
\Delta_r=\delta_r(u)-\delta_r(v).
\tag{D.13}
$$

对组合分数同样记$m_N=s_N(u)-s_N(v)$等。于是

$$
\begin{aligned}
m_N&=m_c+\alpha(m_c-m_r),\\
m_G&=m_N+\Delta_c+\alpha(\Delta_c-\Delta_r),\\
m_{\mathrm{IP}}&=m_N+\Delta_c
=m_G+\alpha(\Delta_r-\Delta_c).
\end{aligned}
\tag{D.14}
$$

当$\alpha>0$且$\Delta_r>\Delta_c$时，参考条件中的额外指令响应使带引导VCD的弃权边际减少$\alpha(\Delta_r-\Delta_c)$。当$u,v\in\mathcal S_g$时，由引理D.1可得归一化概率的精确关系

$$
\frac{p_{\mathrm{IP}}(u)/p_{\mathrm{IP}}(v)}{p_G(u)/p_G(v)}
=\exp\{\alpha(\Delta_r-\Delta_c)\}.
\tag{D.15}
$$

因此，两候选之间由答案领先变为弃权领先的严格条件为

$$
m_G<0<m_G+\alpha(\Delta_r-\Delta_c).
\tag{D.16}
$$

若讨论整个候选集合中的贪心胜出者，令$k(w)=\alpha[\delta_r(w)-\delta_c(w)]$。候选$u$成为IP-VCD的唯一最大分数词元，当且仅当

$$
s_G(u)-s_G(v)+k(u)-k(v)>0,
\qquad\forall v\in\mathcal S_g\setminus\{u\}.
\tag{D.17}
$$

式（D.16）描述指定候选对的翻转，式（D.17）描述全词表保留集合中的胜出条件。两者分别对应局部边际记录与实际argmax记录。

无引导参考也可以改变弃权的相对支持：由式（D.14），$m_r>m_c$且$\alpha>0$时，原生对比相对正常分布的边际改变量为$\alpha(m_c-m_r)<0$。这项视觉差分继续包含在IP-VCD中；式（D.12）精确列出IP-VCD对额外指令交互的处理。

### D.2.4 仅移除参考引导的对照

仅在参考侧移除弃权条件句，其分数为

$$
\begin{aligned}
s_{\mathrm{ref\text{-}off}}
&=(1+\alpha)g-\alpha r\\
&=s_N+(1+\alpha)\delta_c
=s_{\mathrm{IP}}+\alpha\delta_c.
\end{aligned}
\tag{D.18}
$$

该对照保留了$1+\alpha$倍的正常指令增量。IP-VCD增加的无引导正常条件$c$，使视觉差分的系数$\alpha$与正常指令的单位系数分别确定。

## D.3 指令保持性质与三条件系数

### D.3.1 相对指令作用的精确保持

**定理D.1（单位指令增量保持）。** 固定生成前缀，并在同一个非空支持$S$上分别归一化$s_N$和$s_{\mathrm{IP}}$。对任意$u,v\in S$，有

$$
\boxed{
\log\frac{\mathcal P_S[s_{\mathrm{IP}}](u)}{\mathcal P_S[s_{\mathrm{IP}}](v)}
-
\log\frac{\mathcal P_S[s_N](u)}{\mathcal P_S[s_N](v)}
=\delta_c(u)-\delta_c(v).
}
\tag{D.19}
$$

**证明。** 由引理D.1，左侧等于

$$
\begin{aligned}
&[s_{\mathrm{IP}}(u)-s_{\mathrm{IP}}(v)]-[s_N(u)-s_N(v)]\\
&=[s_N(u)+\delta_c(u)-s_N(v)-\delta_c(v)]-[s_N(u)-s_N(v)]\\
&=\delta_c(u)-\delta_c(v).
\end{aligned}
$$

证毕。对于各自采用$\mathcal S_c$和$\mathcal S_g$的实际分布，同一结论适用于$u,v\in\mathcal S_c\cap\mathcal S_g$。对新进入$\mathcal S_g$的候选，采用$s_N$在$\mathcal S_g$上的受限归一化，可以单独观察同一候选集合中的指令作用。

将式（D.19）指数化，有

$$
\frac{\mathcal P_S[s_{\mathrm{IP}}](u)}{\mathcal P_S[s_{\mathrm{IP}}](v)}
=
\frac{\mathcal P_S[s_N](u)}{\mathcal P_S[s_N](v)}
\frac{p_g(u)/p_c(u)}{p_g(v)/p_c(v)}.
\tag{D.20}
$$

因此，正常图像上由指令引起的候选优势比，完整乘入原生视觉对比的候选优势比中。对应地，相对带引导正常基础$p_g$，IP-VCD施加的视觉增量为

$$
\log\frac{p_{\mathrm{IP}}(u)}{p_{\mathrm{IP}}(v)}
-\log\frac{p_g(u)}{p_g(v)}
=\alpha[d(u)-d(v)],\qquad u,v\in\mathcal S_g.
\tag{D.21}
$$

式（D.19）与式（D.21）共同表达两项职责：正常指令以单位系数进入候选竞争，视觉差分以$\alpha$为强度进行调整。

### D.3.2 三条件线性组合中的唯一系数

**定理D.2（三条件线性族的系数刻画）。** 考虑与词元无关的系数组成的三条件线性族

$$
s_{a,b,e}=a g+b c+e r.
\tag{D.22}
$$

要求该族对所有正条件分布满足以下两项性质。其一，固定$c,r$而将$g$替换为$g'$时，任意候选对的分数改变量等于该候选对在$g'-g$中的改变量。其二，在$g=c$时，候选对分数差恢复为原生VCD的分数差。则系数唯一为

$$
(a,b,e)=(1,\alpha,-\alpha).
\tag{D.23}
$$

**证明。** 第一项要求给出

$$
a\big[(g'(u)-g(u))-(g'(v)-g(v))\big]
=(g'(u)-g(u))-(g'(v)-g(v)).
$$

正分布的候选对对数优势比可以连续变化，因此该恒等式要求$a=1$。第二项要求在$g=c$时，对任意$c,r,u,v$满足

$$
(a+b)[c(u)-c(v)]+e[r(u)-r(v)]
=(1+\alpha)[c(u)-c(v)]-\alpha[r(u)-r(v)].
$$

分别变化$c$与$r$的候选对优势比，得到$a+b=1+\alpha$及$e=-\alpha$。代入$a=1$，即得$b=\alpha$。证毕。

唯一性针对式（D.22）的线性族及上述两项设计要求。所有分数共同增加一个与词元无关的常数时，解码分布保持相同。

### D.3.3 概率形式与退化情形

将三条件分数代入受限归一化，得到

$$
p_{\mathrm{IP}}(v)
=\frac{\mathbf1\{v\in\mathcal S_g\}\,p_g(v)
\left[p_c(v)/p_r(v)\right]^\alpha}
{\sum_{w\in\mathcal S_g}p_g(w)\left[p_c(w)/p_r(w)\right]^\alpha}.
\tag{D.24}
$$

这种基础分布与似然比相乘的形式属于解码期专家乘积构造[2]。这里的三个分布均由同一冻结模型在已明确的三个输入条件下产生。

若$g=c$，则$\mathcal S_g=\mathcal S_c$，分数和支持共同恢复原生VCD。若$c=r$或$\alpha=0$，则

$$
p_{\mathrm{IP}}(v)
=\frac{\mathbf1\{v\in\mathcal S_g\}p_g(v)}{p_g(\mathcal S_g)}.
\tag{D.25}
$$

此时得到引导基础在保留集合上的条件分布。由于$\mathcal S_g$包含全部最大概率候选，贪心选择与引导Direct相同。若$\beta=0$，式（D.25）进一步等于完整的$p_g$。

## D.4 KL正则化下的唯一最优分布

### D.4.1 目标函数与闭式解

固定生成前缀及支持$S=\mathcal S_g$，记引导基础在该支持上的条件分布为

$$
G(v)=\frac{p_g(v)}{p_g(S)},\qquad v\in S,
\tag{D.26}
$$

并令$\Delta(S)$表示支持$S$上的概率单纯形。视觉相对支持为$d(v)=c(v)-r(v)$。考虑局部目标

$$
\mathcal L_\alpha(\pi)
=\alpha\sum_{v\in S}\pi(v)d(v)
-\sum_{v\in S}\pi(v)\log\frac{\pi(v)}{G(v)},
\qquad\pi\in\Delta(S),
\tag{D.27}
$$

其中约定$0\log0=0$。第一项奖励正常视觉相对参考视觉的支持，第二项以KL散度约束相对引导基础的变化。

**定理D.3（固定支持上的变分最优性）。** 目标（D.27）的唯一最大化分布为

$$
\pi_\alpha^*(v)
=\frac{G(v)e^{\alpha d(v)}}{\overline Z_\alpha},
\qquad
\overline Z_\alpha=\sum_{w\in S}G(w)e^{\alpha d(w)}.
\tag{D.28}
$$

该分布恰为$p_{\mathrm{IP}}$。

**证明。** 由式（D.28），

$$
\log\pi_\alpha^*(v)=\log G(v)+\alpha d(v)-\log\overline Z_\alpha.
$$

对任意$\pi\in\Delta(S)$，代入KL散度：

$$
\begin{aligned}
D_{\mathrm{KL}}(\pi\|\pi_\alpha^*)
&=\sum_v\pi(v)[\log\pi(v)-\log G(v)-\alpha d(v)+\log\overline Z_\alpha]\\
&=D_{\mathrm{KL}}(\pi\|G)-\alpha\mathbb E_\pi[d]+\log\overline Z_\alpha.
\end{aligned}
$$

整理得

$$
\boxed{
\mathcal L_\alpha(\pi)
=\log\overline Z_\alpha-D_{\mathrm{KL}}(\pi\|\pi_\alpha^*).
}
\tag{D.29}
$$

KL非负性可直接由$\log x\leq x-1$推出。对$\pi(v)>0$的项求和，有

$$
-\sum_{v:\pi(v)>0}\pi(v)\log\frac{\pi_\alpha^*(v)}{\pi(v)}
\geq\sum_{v:\pi(v)>0}[\pi(v)-\pi_\alpha^*(v)]\geq0.
$$

等号要求全部正概率位置的比值为1，且$\pi_\alpha^*$在其余位置的总质量为0。由于$\pi_\alpha^*$在$S$上处处为正，等号恰在$\pi=\pi_\alpha^*$时成立。因此（D.29）的唯一最大值为$\log\overline Z_\alpha$，由$\pi_\alpha^*$取得。再将$G=p_g/p_g(S)$代入（D.28），公共因子$p_g(S)$抵消，得到（D.24）。证毕。

这一最优化解释将IP-VCD对应到固定前缀上的明确目标。用于实验的联合决策效用采用完整回答及冻结参考标签计算，两者分别承担局部构造与最终行为评价的职责。

### D.4.2 对比强度怎样改变分布

令$\psi(\alpha)=\log\overline Z_\alpha$。有限词表允许逐项求导，得到

$$
\psi'(\alpha)=\mathbb E_{\pi_\alpha^*}[d],\qquad
\psi''(\alpha)=\operatorname{Var}_{\pi_\alpha^*}(d)\geq0.
\tag{D.30}
$$

**证明。** 对$\overline Z_\alpha$求导并除以自身，得到第一式。继续求导，第二式为$\mathbb E[d^2]-(\mathbb E[d])^2$。等价地，对每个候选有

$$
\frac{\partial\pi_\alpha^*(v)}{\partial\alpha}
=\pi_\alpha^*(v)\big[d(v)-\mathbb E_{\pi_\alpha^*}[d]\big].
\tag{D.31}
$$

证毕。因此，在固定四路分布与固定支持下，增大$\alpha$提高视觉相对支持高于当前均值的候选概率，并使其期望单调不减。

对预先定义的候选组$A\subset S$，令$B=S\setminus A$，且两组非空。求和得到

$$
\frac{\partial\pi_\alpha^*(A)}{\partial\alpha}
=\pi_\alpha^*(A)\pi_\alpha^*(B)
\left[\mathbb E_{\pi_\alpha^*}[d\mid A]-\mathbb E_{\pi_\alpha^*}[d\mid B]\right].
\tag{D.32}
$$

候选组的概率增长由两组的条件平均视觉支持之差决定，这给出了对比强度作用于整组候选的精确条件。

## D.5 从单个候选到候选组概率

### D.5.1 指数倾斜的一般分组恒等式

令$p$是$\mathcal V$上的正分布，$A,B$构成$\mathcal V$的非空划分，且$A\cap S$、$B\cap S$均非空。对任意有限函数$f$，定义

$$
q(v)=\frac{\mathbf1\{v\in S\}p(v)e^{f(v)}}{\sum_{w\in S}p(w)e^{f(w)}}.
\tag{D.33}
$$

记$O_p(A)=p(A)/p(B)$，并令$p_S=p(\cdot\mid S)$。有

$$
\boxed{
\begin{aligned}
\log O_q(A)-\log O_p(A)
={}&\log\frac{p(S\mid A)}{p(S\mid B)}\\
&+\log\mathbb E_{p_S(\cdot\mid A)}[e^f]
-\log\mathbb E_{p_S(\cdot\mid B)}[e^f].
\end{aligned}}
\tag{D.34}
$$

**证明。** 对两组分别求和，归一化常数相消：

$$
\begin{aligned}
\frac{q(A)}{q(B)}
&=\frac{\sum_{v\in A\cap S}p(v)e^{f(v)}}{\sum_{v\in B\cap S}p(v)e^{f(v)}}\\
&=\frac{p(A)p(S\mid A)}{p(B)p(S\mid B)}
\frac{\mathbb E_{p(\cdot\mid A\cap S)}[e^f]}{\mathbb E_{p(\cdot\mid B\cap S)}[e^f]}.
\end{aligned}
$$

取对数即得（D.34）。证毕。第一项刻画候选截断改变了多少组间优势比，其余两项刻画保留候选上的重新加权。若$A\cap S$为空，则$q(A)=0$；若$B\cap S$为空，则$q(A)=1$，相应概率可直接从支持确定。

对IP-VCD，取$p=p_g$、$f=\alpha(c-r)$及$S=\mathcal S_g$，式（D.34）即给出任意登记弃权词元组相对引导正常分布的完整概率变化。

### D.5.2 移除参考交互后的组概率变化

带引导VCD与IP-VCD共享$\mathcal S_g$。令$k=\alpha(\delta_r-\delta_c)$，由式（D.12）有

$$
p_{\mathrm{IP}}(v)
=\frac{p_G(v)e^{k(v)}}{\mathbb E_{p_G}[e^k]},\qquad v\in\mathcal S_g.
\tag{D.35}
$$

因此，对保留集合中的两组$A,B$，

$$
\log O_{p_{\mathrm{IP}}}(A)-\log O_{p_G}(A)
=\log\mathbb E_{p_G(\cdot\mid A)}[e^k]
-\log\mathbb E_{p_G(\cdot\mid B)}[e^k].
\tag{D.36}
$$

**证明。** 将$s_{\mathrm{IP}}=s_G+k$代入受限softmax，分母除以$Z_{\mathcal S_g}(s_G)$，得到（D.35）；随后使用（D.34）在同一支持上的形式即可。证毕。

组概率严格增加的充要条件为

$$
p_{\mathrm{IP}}(A)>p_G(A)
\iff
\mathbb E_{p_G(\cdot\mid A)}[e^k]
>\mathbb E_{p_G(\cdot\mid B)}[e^k].
\tag{D.37}
$$

一个直观的充分条件是$\min_{u\in A}k(u)>\max_{v\in B}k(v)$。对于单个候选，精确条件为$k(u)>\log\mathbb E_{p_G}[e^k]$。这些关系将局部加分与归一化后的概率变化连接起来：提高组间优势需要比较两组候选的加权增量。

### D.5.3 VCD的候选支持、组间偏好与组内分配分解

令$p,q$分别为某一VCD实例的正常与参考概率，支持为$S$，对比分布为

$$
m(v)\propto\mathbf1\{v\in S\}p(v)^{1+\alpha}q(v)^{-\alpha}.
\tag{D.38}
$$

定义$p_S=p(\cdot\mid S)$、$q_S=q(\cdot\mid S)$，并在两个保留组上定义$p_A=p_S(\cdot\mid A)$、$q_A=q_S(\cdot\mid A)$及$p_B,q_B$。对$\rho>1$，Rényi散度采用[3]的有限分布定义

$$
D_\rho(P\|Q)=\frac{1}{\rho-1}\log\sum_v P(v)^\rho Q(v)^{1-\rho}.
\tag{D.39}
$$

**命题D.2（整组对比作用的精确分解）。** 当$\alpha>0$且两组在$S$上都有正质量时，

$$
\boxed{
\begin{aligned}
\log O_m(A)-\log O_p(A)
={}&\underbrace{\log O_{p_S}(A)-\log O_p(A)}_{\text{候选支持作用}}\\
&+\underbrace{\alpha[\log O_{p_S}(A)-\log O_{q_S}(A)]}_{\text{参考组间偏好作用}}\\
&+\underbrace{\alpha[D_{1+\alpha}(p_A\|q_A)-D_{1+\alpha}(p_B\|q_B)]}_{\text{组内候选分配作用}}.
\end{aligned}}
\tag{D.40}
$$

**证明。** 将$p,q$替换为$p_S,q_S$只引入一个组间相同的比例常数。组$A$上的未归一化质量可写为

$$
\begin{aligned}
F_A
&=\sum_{v\in A\cap S}p_S(v)^{1+\alpha}q_S(v)^{-\alpha}\\
&=p_S(A)^{1+\alpha}q_S(A)^{-\alpha}
\sum_{v\in A\cap S}p_A(v)^{1+\alpha}q_A(v)^{-\alpha}\\
&=p_S(A)^{1+\alpha}q_S(A)^{-\alpha}
\exp\{\alpha D_{1+\alpha}(p_A\|q_A)\}.
\end{aligned}
\tag{D.41}
$$

对$B$同理。由$O_m(A)=F_A/F_B$，得

$$
\log O_m(A)
=(1+\alpha)\log O_{p_S}(A)-\alpha\log O_{q_S}(A)
+\alpha[D_{1+\alpha}(p_A\|q_A)-D_{1+\alpha}(p_B\|q_B)].
$$

减去$\log O_p(A)$，再拆分$(1+\alpha)\log O_{p_S}(A)$即得（D.40）。证毕。当$\alpha=0$时，$m=p_S$，概率变化由候选支持项给出。

该分解将整组弃权变化落实到三个可计算对象：哪些候选通过阈值、参考分布怎样偏向两个组，以及各组内部怎样分配不同表达的概率。当保留集合只含两个候选、每组各一个时，两个组内散度均为零，式（D.40）恢复候选对的边际分解。局部弃权组由登记词元集合确定；完整回复的语义弃权使用下一节的终止序列事件定义。

## D.6 自回归生成中的完整回复概率

### D.6.1 局部归一化沿路径累积

设$\mathcal Y_T$为在首个登记终止词元处结束、或到达最大长度$T$时结束的所有输出词元序列。对任意方法$M$，其逐步归一化分数诱导

$$
P_M(y)=\prod_{t=1}^{|y|}p_{M,t}(y_t\mid y_{<t}),\qquad y\in\mathcal Y_T.
\tag{D.42}
$$

生成树的每个内部结点将其概率质量按归一化的下一词元概率分配给子结点，因此所有叶结点的概率和为1。对于由终止词元结束的回复，乘积包含该终止词元；对于达到长度上限的回复，乘积包含实际生成的最后一个词元。

对IP-VCD，定义

$$
Z_{\mathrm{IP},t}(y_{<t})
=\sum_{v\in\mathcal S_{g,t}(y_{<t})}
\exp\{g_t(v)+\alpha[c_t(v)-r_t(v)]\}.
\tag{D.43}
$$

对每一步均通过支持约束的路径，代入（D.42）并取对数：

$$
\begin{aligned}
\log P_{\mathrm{IP}}(y)
={}&\sum_t g_t(y_t)
+\alpha\sum_t[c_t(y_t)-r_t(y_t)]\\
&-\sum_t\log Z_{\mathrm{IP},t}(y_{<t}).
\end{aligned}
\tag{D.44}
$$

路径一旦包含支持外词元，其$P_{\mathrm{IP}}(y)=0$。各条件分布在式（D.44）中均沿同一条$y$计算，局部归一化常数也由该路径的前缀确定。

进一步令$P_g(y)=\prod_t e^{g_t(y_t)}$、$P_c(y)=\prod_t e^{c_t(y_t)}$、$P_r(y)=\prod_t e^{r_t(y_t)}$，并令$\chi_g(y)=\prod_t\mathbf1\{y_t\in\mathcal S_{g,t}(y_{<t})\}$。固定$\alpha$时，完整乘积形式为

$$
P_{\mathrm{IP}}(y)
=\frac{\chi_g(y)P_g(y)[P_c(y)/P_r(y)]^\alpha}
{\prod_t Z_{\mathrm{IP},t}(y_{<t})}.
\tag{D.45}
$$

式（D.45）保留了每个前缀对应的分母。实际回复重放逐步记录这些量，便能把局部分数调整累计为完整序列概率。

### D.6.2 IP-VCD与带引导VCD的路径差异

记带引导VCD对应的逐步归一化常数为$Z_{G,t}$。两个方法在同一给定前缀上使用相同的$\mathcal S_{g,t}$。对共同正概率路径，由（D.12）与（D.44）有

$$
\begin{aligned}
\log\frac{P_{\mathrm{IP}}(y)}{P_G(y)}
={}&\alpha\sum_t[\delta_{r,t}(y_t)-\delta_{c,t}(y_t)]\\
&-\sum_t\log\frac{Z_{\mathrm{IP},t}(y_{<t})}{Z_{G,t}(y_{<t})}.
\end{aligned}
\tag{D.46}
$$

**证明。** 对每一步写出$\log p_{\mathrm{IP},t}-\log p_{G,t}$，利用分数差（D.12），再沿路径求和。证毕。

式（D.46）将参考交互的移除量与每一步重新分配概率质量的作用同时计入。贪心路径从首次分岔后形成各自的前缀，完整路径比较通过逐条教师强制重放计算上述概率。

### D.6.3 完整弃权事件的概率变化

令$\mathcal A\subset\mathcal Y_T$为按完整回复语义标记的弃权事件，$\mathcal B=\mathcal Y_T\setminus\mathcal A$。取在$\mathcal Y_T$上为正的基础路径分布$P_0$，例如未截断的引导条件分布$P_g$；取另一归一化路径分布$P_1$。定义路径的对数似然变化

$$
D(y)=\log\frac{P_1(y)}{P_0(y)},
\qquad e^{D(y)}=0\ \text{当}\ P_1(y)=0.
\tag{D.47}
$$

只要两个事件在所比较的分布上均有正质量，就有

$$
\boxed{
\log\frac{P_1(\mathcal A)}{P_1(\mathcal B)}
-\log\frac{P_0(\mathcal A)}{P_0(\mathcal B)}
=\log\mathbb E_{P_0(\cdot\mid\mathcal A)}[e^D]
-\log\mathbb E_{P_0(\cdot\mid\mathcal B)}[e^D].
}
\tag{D.48}
$$

**证明。** 由$P_1(y)=P_0(y)e^{D(y)}$，对事件$\mathcal A$求和得到

$$
P_1(\mathcal A)=P_0(\mathcal A)\mathbb E_{P_0(\cdot\mid\mathcal A)}[e^D].
$$

对$\mathcal B$同理，两式取比值并取对数即得。证毕。

若实际重放使用登记的有限完整回复集合$\mathcal C\subseteq\mathcal Y_T$，分别构造$P_j(\cdot\mid\mathcal C)$，并以$\mathcal A\cap\mathcal C$和$\mathcal B\cap\mathcal C$分组，则（D.48）同样成立。每条候选记录保存实际词元序列、结束方式及其完整语义标签，相同词元路径按一次计数。这样，单个词元的准入、指定候选对的翻转和完整弃权事件分别在各自的样本空间中具有明确的计算表达。

## D.7 向IP-M3ID推广

M3ID使用带视觉条件与去图像条件之间的差分，并通过下一词元置信度和时间权重控制差分强度[4]。在IP-M3ID中，$c_t$继续表示无弃权引导的正常视觉对数概率，$r_t$改为保留相同文本及生成前缀的去图像对数概率。令$\tau_t$为实现登记的时间索引，置信度阈值为$\kappa$，则当前构造的权重可写为

$$
w_t^c
=\mathbf1\{\max_v e^{c_t(v)}<\kappa\}
\big(e^{\lambda\tau_t}-1\big),
\qquad
\lambda\geq0,\quad\tau_t\geq0.
\tag{D.49}
$$

本研究按登记位置计算权重：若生成循环采用零起始索引$j=t-1$，则$\tau_t=j+k_0$，$k_0$为当前输入无弃权引导任务文本的登记词元长度，因此可随问题改变。采用$\lambda=0.02$、$\kappa=0.3$。权重由$c_t$及该索引决定。在固定前缀上，它对所有候选相同。

IP-M3ID及其同权重无引导形式分别为

$$
\begin{aligned}
s_{\mathrm{IPM},t}&=g_t+w_t^c(c_t-r_t),\\
s_{\mathrm{NM},t}&=c_t+w_t^c(c_t-r_t),\\
s_{\mathrm{IPM},t}-s_{\mathrm{NM},t}&=\delta_{c,t}.
\end{aligned}
\tag{D.50}
$$

本文的IP-M3ID分布定义在完整词表上归一化（等价于不施加VCD候选截断）。于是定理D.1的候选对保持性质对全词表成立；定理D.3在每个固定前缀上以$w_t^c$替换$\alpha$成立。路径概率则按每一步实际权重保留$\sum_t w_t^c[c_t(y_t)-r_t(y_t)]$。

对带引导M3ID，令$w_t^g=\mathbf1\{\max_v e^{g_t(v)}<\kappa\}(e^{\lambda\tau_t^g}-1)$，其中$\tau_t^g$为该引导条件的登记位置；以带引导去图像分布$h_t$作参考，则

$$
s_{\mathrm{GM},t}=g_t+w_t^g(g_t-h_t).
$$

利用$g_t-h_t=(c_t-r_t)+(\delta_{c,t}-\delta_{r,t})$，可得完整差值

$$
\boxed{
s_{\mathrm{IPM},t}-s_{\mathrm{GM},t}
=(w_t^c-w_t^g)(c_t-r_t)
+w_t^g(\delta_{r,t}-\delta_{c,t}).
}
\tag{D.51}
$$

第一项来自实际门控与时间权重的差异，第二项来自参考指令交互。取$w_t^c=w_t^g$时，公式恢复同权重的指令交互分解。该形式允许在保持真实调度规则的同时，分别核算两项来源。

## D.8 生牛肉片案例的逐项代入

取正文LLaVA-Mistral的生牛肉片案例，在首个生成位置比较弃权词元“UN”与答案词元“To”。保持实际输入、前缀和噪声不变，已有四路记录给出以下弃权减答案边际。

| 条件 | $m_c$ | $m_r$ | $m_g$ | $m_h$ |
|---|---:|---:|---:|---:|
| 对数概率边际 | $-12.5625$ | $-11.546875$ | $1.4375$ | $3.0625$ |

由式（D.13），

$$
\begin{aligned}
\Delta_c&=m_g-m_c=1.4375-(-12.5625)=14,\\
\Delta_r&=m_h-m_r=3.0625-(-11.546875)=14.609375.
\end{aligned}
\tag{D.52}
$$

在$\alpha=1$下，原生视觉对比的边际为

$$
m_N=2m_c-m_r=-13.578125.
\tag{D.53}
$$

带引导VCD与IP-VCD分别给出

$$
\begin{aligned}
m_G&=m_N+\Delta_c+(\Delta_c-\Delta_r)\\
&=-13.578125+14-0.609375=-0.1875,\\
m_{\mathrm{IP}}&=m_N+\Delta_c=-13.578125+14=0.421875.
\end{aligned}
\tag{D.54}
$$

两个候选均在引导支持内，因而其优势比变化为$\exp(0.609375)$。该位置的全词表最大分数词元也从“To”变为“UN”，对应真实完整回复“Toast”与“UNKNOWN”。

进一步在共同支持上插入诊断参数$\eta\in[0,1]$，定义

$$
s_\eta=s_{\mathrm{IP}}+\eta\alpha(\delta_c-\delta_r),
\qquad
m_\eta=0.421875-0.609375\eta.
\tag{D.55}
$$

固定前缀、共同支持上候选对分数的精确交叉点为

$$
\eta_* = \frac{0.421875}{0.609375}=\frac9{13}\approx0.692308.
\tag{D.56}
$$

$\eta<9/13$时，该候选对中弃权领先；$\eta>9/13$时，答案领先。已有的$\eta=0,0.5,1$诊断分别选择“UN”“UN”“To”。这一实例将正常指令增量、参考额外响应、共同支持上的分数翻转与实际回复路径连接起来。

## 参考文献

[1] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. 2024. *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*. CVPR, 13872–13882.

[2] Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A. Smith, and Yejin Choi. 2021. *DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts*. ACL-IJCNLP, 6691–6706. DOI: 10.18653/v1/2021.acl-long.522.

[3] Tim van Erven and Peter Harremoës. 2014. *Rényi Divergence and Kullback–Leibler Divergence*. IEEE Transactions on Information Theory, 60(7):3797–3820. DOI: 10.1109/TIT.2014.2320500.

[4] Alessandro Favero, Luca Zancato, Matthew Trager, Siddharth Choudhary, Pramuditha Perera, Alessandro Achille, Ashwin Swaminathan, and Stefano Soatto. 2024. *Multi-Modal Hallucination Control by Visual Information Grounding*. CVPR.