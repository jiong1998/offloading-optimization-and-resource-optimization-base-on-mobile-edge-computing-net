# offloading-optimization-and-resource-optimization-base-on-mobile-edge-computing-net
基于边缘计算网络的卸载优化及资源优化

考虑了一个多用户单窃听者的移动边缘计算(MEC)网络，通过使用深度强化学习算法，并结合凸优化，对MEC网络进行了卸载优化与资源优化。

仿真结果如下所示（部分）：

![max](https://user-images.githubusercontent.com/77431730/221493000-b2f0a5cd-3c26-48c4-ad89-869bf80f934d.jpg)
上图展示的是窃听者共谋的结果。从图中可以明显看出，仅仅跑了100epoch，DQN所跑出来的结果明显优于全本地与全卸载。并且趋近于收敛趋势（笔者自己跑的epoch太少了，跑500个epoch应该就能收敛了）。说明DQN从网络中学习到了相关知识，并将网络性能优化。
![sum](https://user-images.githubusercontent.com/77431730/221493229-ee7f7582-697d-4777-be5c-8606b959b3e0.jpg)
上图展示的是窃听者非共谋的结果。由于非共谋的窃听者会对网络环境产生较大的干扰，所以从途中可以明显看出此图较上图而言存在着明显的波动。但是在如此大干扰情况下，DQN仍能从中学习到知识，并将网络性能优化。

![traditon_compare1](https://user-images.githubusercontent.com/77431730/221493751-227b3460-9bf4-4cbc-99c2-6f49b6ce2df8.jpg)
最后一张图展示的是，结合了传统的MEC优化算法，在这过程中，我们所提出的DQN分配卸载策略，凸优化解决资源分配的算法仍比传统的MEC优化算法更优。这说明我们所提出的算法相较于传统优化算法有着更好的优势

系统模型图如下所示：
<img width="970" alt="fig1" src="https://user-images.githubusercontent.com/77431730/221492746-75708f4a-5c57-4ab3-ab20-81aa21c43cf5.png">

DQN的结构如下两张图所示：

![fig2](https://user-images.githubusercontent.com/77431730/221492887-2c015335-c328-42f6-a2b8-cfab977f744b.png)

![fig3](https://user-images.githubusercontent.com/77431730/221492906-7e82e9a6-dd97-454d-aca7-7c4b67f6b7df.png)

<img width="738" alt="image" src="https://user-images.githubusercontent.com/77431730/221494658-d77fc4eb-8788-4fff-a9ab-120f30153472.png">
