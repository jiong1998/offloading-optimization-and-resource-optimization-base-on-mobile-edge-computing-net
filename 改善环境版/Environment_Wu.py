import math

import numpy
import numpy as np


class Environment(object):

    def __init__(self, user_num, cap_num, W, f_local, omega,
                 F_cap, p_local, p_tran, p_noise, Hn, Hn_e, user_l, Mb_to_bit, lamda_):

        self.user_num = user_num
        self.cap_num = cap_num
        # self.name = name  # 进程名字

        self.f_local = f_local
        self.omega = omega

        # 获取信道参数
        self.Hn = Hn
        self.Hn_e = Hn_e
        self.Hn_A = self.Hn[0]
        self.Hn_B = self.Hn[1]

        # W为总带宽，[capA,capB]，需要取出来
        self.W = W


        # F_cap为A和B总算力（优化对象），[CAPA,CAPB]，需分别均分给n个设备
        self.F_cap = F_cap.reshape(1, -1)  # (2,1)(F_A,F_B) （10*10e8,3*10e8）
        self.F = (np.r_[self.F_cap, self.F_cap] / 2)
        self.F_constant = self.F  # [[5,1,5],[5,1.5]]


        # 传输功率 下个版本也许要利用解析解优化 如p_{i,j}
        self.p_tran = (p_tran / self.user_num).reshape(-1, 1)
        self.P = np.c_[self.p_tran, self.p_tran]
        self.p_tran_constant = self.p_tran  # [[1, 1], [1.5, 1.5]]
        self.p_tran_total = p_tran.reshape(1, -1)  # 该参数仅用于传入神经网络寻

        self.F_P_total = np.append(self.F_cap, self.p_tran_total, axis=1)

        self.p_noise = p_noise
        self.p_local = p_local

        self.omega = omega
        self.Mb_to_bit = Mb_to_bit
        self.lamda_ = lamda_

        self.user_l = user_l  # (2,)

        # 卸载率权重 和卸载率的初始化(归一化)
        self.off_rate_weight = self.init_offloading_weight()  # 卸载率权重(未归一化)
        self.offload_rate = np.zeros((self.user_num, self.cap_num + 1))
        self.off_rate_weight_normalization()#卸载率归一化

        # 环境状态
        self.state = self.update_state()  # (6,1)

        self.tc_pre = self.get_total_cost()
        self.tc_cur = 0.0

        self.total_local_latency = self.tc_pre # 本地
        self.total_cap_offload_latency = self.get_all_cap_latency_cost()  # 全卸载
        self.done = False

        # if name == 'w00':
        #     print("all cap_cost{} all_local_cost{}".format(self.total_cap_offload_latency, self.total_local_latency))

    def init_offloading_weight(self):
        offload_weight = np.zeros((self.user_num, self.cap_num + 1))
        # e.g
        # offloading ={[1,0,0],[1,0,0]}
        for i in range(self.user_num):
            for j in range(self.cap_num+1):
                if j ==0:
                    offload_weight[i][j] = 1.0
                else:
                    offload_weight[i][j] = 0.1  # 全卸载在优化算力会出现问题 分母为0
        return offload_weight

    def reset(self):
        self.off_rate_weight = self.init_offloading_weight()
        self.off_rate_weight_normalization()


        # s = np.append(self.F_P_total, self.state.reshape(1, -1), axis=1) 加入总功率和总算力self.F_cap，self.p_tran_total

        return self.state

    def action_state_num(self):
        action_n = self.user_num * (self.cap_num + 1)  # dqn
        state_num = self.user_num * (self.cap_num + 1)
        return action_n, state_num

    def step(self, a, cur_step, ):
        # a [0,1,2,3,4,5]中一个
        # [0,1,2] and [3,4,5]分别对应user_1 user2的动作
        # 对应a的卸载率
        # 总动作数量 = (cap_num+1)*user_num

        # 1.0 更新卸载率权重
        #  w =  (cur_step / 100)+ 1.0

        w = cur_step // 50 + 1.0

        step = w * 1.0

        user_index = a // (self.cap_num + 1)#选出是哪个用户
        action_index = a % (self.user_num + 1)#选出是本地还是capA，capB

        self.off_rate_weight[user_index][action_index] += step

        # 1.1 更新卸载率
        self.off_rate_weight_normalization()

        # 2. 更新环境状态
        self.state = self.update_state()

        # 3.1 优化P和F
        self.optimaztion_for_power_and_F_Cap()
        self.tc_cur = self.get_total_cost()

        if self.tc_cur > self.tc_pre:
            reward = -100
        elif self.tc_cur < self.tc_pre:
            reward = 100
        else:
            reward = -50

        # 4.缓存总代价
        self.tc_pre = self.tc_cur

        # s = np.append(self.F_P_total, self.state.reshape(1,-1), axis=1) 加入总功率和总算力self.F_cap，self.p_tran_total


        return self.state, reward, self.total_local_latency, self.tc_cur, self.total_cap_offload_latency, self.done#self.tc_cur为传统方法

    def init_P_and_F(self):
        self.F = np.array([[5e9, 1.5e9], [5e9, 1.5e9]])
        self.P = np.array([[1., 1.], [1.5, 1.5]])

    def update_state(self):#更新环境
        temp = np.zeros_like(self.offload_rate)

        for i in range(len(self.user_l)):
            temp[i] = self.offload_rate[i] * self.user_l[i]

        state = np.r_[temp[0], temp[1]].reshape(-1, 1).squeeze()
        return state

    def off_rate_weight_normalization(self):
        for i in range(self.user_num):
            total = np.sum(self.off_rate_weight[i])
            for j in range(self.cap_num + 1):
                self.offload_rate[i][j] = self.off_rate_weight[i][j] / total

    def get_all_cap_latency_cost(self):
        # 全卸载
        temp = self.offload_rate
        self.offload_rate = np.array([[0., 0.5, 0.5], [0., 0.5, 0.5]])
        all_cap_cost = self.get_total_cost()
        self.offload_rate = temp
        return all_cap_cost

    def get_all_local_latency(self):
        # 全本地计算
        temp = self.offload_rate
        self.offload_rate = np.array([[1, .0, .0], [1., .0, .0]])
        all_cap_cost = self.get_total_cost()
        self.offload_rate = temp
        return all_cap_cost

    def get_total_cost(self):
        a = self.F
        b = self.P
        T_total = self.get_total_latency()
        E_total = self.get_total_energy()
        Cost_total = self.lamda_ * T_total + (1.0 - self.lamda_) * E_total
        return Cost_total

    def get_total_energy(self):

        return 0.0

    def get_total_local_latency(self):
        # 采用∑方式

        sum = 0
        for i in range(self.user_num):
            t_local_i = self.user_l[i] * self.offload_rate[i][0] * self.omega * self.Mb_to_bit / self.f_local[i]
            sum += t_local_i
        return sum



    def get_transmit_latency(self):

        hn_e = self.Hn_e.reshape(-1, 1)  # (2,1)
        hn_e = np.c_[hn_e, hn_e]  # (2,2)

        user_l = self.user_l.reshape(-1, 1)  # ( 2,1)
        user_l = np.c_[user_l, user_l]  # (2,2)

        # R(i,j) -->(2,2)
        # Shannon formula
        R = self.W * np.log2(1 + self.P * self.Hn / (self.p_noise ** 2)) \
            - self.W * np.log2(1 + self.P * hn_e / (self.p_noise ** 2))

        # T_trans = l_{i,j} *a_{i,j} * Mb_to_bit / R_{i,j}
        T_trans = np.multiply(user_l, self.offload_rate[:, 1:]) * self.Mb_to_bit / R
        return T_trans

    def get_cap_latency(self):
        user_l = self.user_l.reshape(-1, 1)  # ( 2,1)
        user_l = np.c_[user_l, user_l]  # (2,2)

        # (2,2)
        T_cap = np.multiply(user_l, self.offload_rate[:, 1:]) * self.Mb_to_bit * self.omega / self.F
        return T_cap

    def get_total_latency(self):
        T_local = self.get_total_local_latency()  # 标量
        T_trans = self.get_transmit_latency()  # (2,2)
        T_cap = self.get_cap_latency()  # (2,2)
        T_total = T_local + T_trans.sum() + T_cap.sum()

        return T_total

    def optimaztion_for_power_and_F_Cap(self):
        hn_e = self.Hn_e.reshape(-1, 1)  # (2,1)
        hn_e = np.c_[hn_e, hn_e]  # (2,2)

        user_l_padding = self.user_l.reshape(-1, 1)
        user_l = np.c_[user_l_padding, user_l_padding]
        offload_rate = self.offload_rate[:, 1:]

        a = self.W * np.log2((self.Hn / hn_e))

        b = (self.W * self.p_noise ** 2 * (1 / (hn_e) - 1 / (self.Hn))) / math.log(2)

        # 求lamda_i
        fenzi = np.sqrt(offload_rate * user_l * b) / a
        # 2个用户直接写死 ，按列加和
        row_temp = fenzi[:, 0] + fenzi[:, 1]
        fenzi = np.c_[(row_temp, row_temp)]

        row_temp = (b / a)[:, 0] + (b / a)[:, 1]
        fenmu = self.p_tran_constant - np.c_[row_temp, row_temp]
        lada = np.square(fenzi / fenmu)

        # 求解mu
        # 按照行加和
        fenzi = np.sqrt(offload_rate * user_l)
        row_temp = fenzi[0, :] + fenzi[1, :]
        fenzi = np.r_[row_temp.reshape(1, -1), row_temp.reshape(1, -1)]
        fenmu = self.F_constant * 2

        mu = np.square(fenzi / fenmu)

        if (lada == 0).any():
            return

        if (mu == 0).any():
            return

        self.P = (np.sqrt(offload_rate * user_l * b / lada) + b) / a
        self.F = np.sqrt(offload_rate * user_l / mu)

        if (offload_rate == 0).any():
            a = 0


