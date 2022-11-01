#%%
import os
# import Writedata
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#2021。 10。27最终版。优化了算力和功率
# 输入的s为：[任务量1✖️卸载率1，任务量2✖️卸载率2， .....]
# 输出的a为某个用户卸载率增加或者减少

#2cap 2user

import torch
import numpy as np

import matplotlib.pyplot as plt
from environment_optimize1 import environment
from environment_tradition import environment1


#%%

# user_num=input("please input user_num")

#初始化参数-----------------
user_num=2
cap_num=2
users_num=int(user_num)

Hn = np.abs(1.1 * np.sqrt(1 / 2) * (np.random.normal(1, 1, [user_num, cap_num]) + np.complex('j') *
                                        np.random.  normal(1, 1, [user_num, cap_num]))) ** 2
Hn_e = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 1, user_num) + np.complex('j') *
                                          np.random.normal(1, 1, user_num))) ** 2##窃听者,[user1,user2]
Hn_0=Hn[0]
Hn_1=Hn[1]

user_l = np.array([[np.random.randint(10,20), np.random.randint(150,200)]])


sus=2000
epochs=100
#----------------
env=environment(user_num=2,cap_num=2,W=[5,5],
                f_local=np.array([[1.4*10e8,0.43*10e8]]),
                omega=330 / 8,
                F_cap=np.array([[10*10e8,3*10e8]]),p_local=1e-1,p_tran=np.array([[2,3]]),lamuda=1,
                noise=0.1,Hn_0=Hn_0,Hn_1=Hn_1,Hn_e=Hn_e,user_l=user_l,
                suspend=sus
)

#改
#获取环境的相关参数:
n_action,n_state=env.action_state_num()
#其他超参数的定义：
MEMORY_CAPACITY=5000#记忆库的存储大小
greedy=0.9#贪婪概率
discount=0.9#折扣率
batch_size=64#一次训练32个数据
TARGET_REPLACE_ITER=10#目标网络的更新频率

#%%

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1=torch.nn.Linear(n_state,20)#(10,20)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2=torch.nn.Linear(20,30)
        self.linear2.weight.data.normal_(0, 0.1)
        self.linear3=torch.nn.Linear(30,18)
        self.linear3.weight.data.normal_(0, 0.1)
        self.linear4=torch.nn.Linear(18,n_action)#(30,12)
        self.linear4.weight.data.normal_(0, 0.1)
        self.relu=torch.nn.ReLU()
    def forward(self,x):
        x=self.relu(self.linear1(x))
        x=self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        q_values=self.linear4(x)
        return q_values


#%%

class DQN ():
    def __init__(self):#可能要改成def __init__(self,n_state,n_action)

        #记忆库：存s,a,r,s_ 一共6维
        self.memory=np.zeros((MEMORY_CAPACITY,n_state*2+2))
        self.memory_count=0#记忆库计数，用于求index，以及当记忆库计数 > 记忆库容量时 开始学习

        #目标网络,参数更新慢。参数更新跟着agent_network走。用于预测q现实
        self.target_network=Net()

        #目标网络更新计数
        self.learn_step_counter=0

        #用于训练的网络，参数更新及时。用于预测q估计
        self.agent_network=Net()

        #优化器与损失器
        self.loss_func=torch.nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.agent_network.parameters(),lr=0.001)
    def choose_action(self,s):#这里每次只输入一个样本。
        # 输入的s为：[任务量1✖️卸载率1，任务量2✖️卸载率2]
        # 输出的action为0或1或2或3
        if np.random.uniform() < greedy:#选择最优动作
            #送入神经网络前，先要处理s成为torch型，且要求shape为(1,4)
            #s=torch.tensor(s,dtype=float).reshape(1,4)
            #s=s.to(torch.float32)
            s=torch.FloatTensor(s).reshape(-1,n_state)

            #把s送入神经网络得到a_value, a_value的shape 为（4，1）。并且得到的是四个动作q值
            action_value=self.agent_network.forward(s)
            #输出每个样本中预测的最大值，输出[最大值，下标]。只要下标
            _,action=torch.max(action_value.data,dim=1)
            #在从numpy转为int值
            action=action.numpy()[0]
            #此时action可能是0、1、2、3


        else:#随机选择动作
            action = np.random.randint(0, n_action)

        return action

    def store_transition(self,s,action,r,s_):
        #将四个值打包在一起

        #action为0or 1 or 2 or 3
        s=s.reshape(n_state)
        s_=s_.reshape(n_state)

        self.transition=np.hstack((s,[action,r],s_))

        #如果记忆库满了, 就覆盖老数据
        index = self.memory_count % MEMORY_CAPACITY

        #传入记忆库

        self.memory[index,:]=self.transition
        self.memory_count+=1
        #print("目前记忆库存储的总数为",self.memory_count)#要删除

    def learn(self):#从记忆库中选择batch个数据进行训练学习
        #首先判断target_network是否需要更新

        if self.learn_step_counter % TARGET_REPLACE_ITER==0:
            self.target_network.load_state_dict(self.agent_network.state_dict())#更新目标网络参数
        self.learn_step_counter+=1

        #随机从memory中抽32个数据，即batch=32

        sample_index=np.random.choice(MEMORY_CAPACITY,batch_size)#从0-1999中抽32个数字，shape为（32，）
        batch_memory=self.memory[sample_index,:]#获取32个数据。 shape 为（32,6）


        b_s = torch.FloatTensor(batch_memory[:, :n_state])
        b_a = torch.LongTensor(batch_memory[:, n_state:n_state+1].astype(int))
        b_r = torch.FloatTensor(batch_memory[:, n_state+1:n_state+2])
        b_s_ = torch.FloatTensor(batch_memory[:, -n_state:])
        # b_memory shape (batch,6)
        # b_s shape (batch,2)
        # b_a shape (batch,1)
        # b_r shape (batch,1)
        # b_s_ shape (batch,2)


        #把样本state送入agent网络训练，得到q估计
        q_predict=self.agent_network(b_s)#shape 为 (32,4)
        q_predict=q_predict.gather(1,b_a)#shaoe 为 (32,1)。根据b_a找出q估计


        # #把样本送入target网络计算，得到Q(s+1)
        # q_next=self.target_network(b_s_).detach()# q_next 不进行反向传播, 所以使用detach。shape为 (32,4)
        # q_next,_=torch.max(q_next,dim=1)#输出每个样本中预测的最大值，输出[最大值，下标]。只要最大值
        # q_next=q_next.reshape((-1,1))#q_next的shape 为（32,1）
        # #计算q真实
        # q_target=b_r+discount*q_next#q_target的shape 为（32,1）

        #改用ddqn
        q_next=self.agent_network(b_s_).detach()#
        _,b_a_=torch.max(q_next,dim=1)#由dqn网络选出动作
        b_a_=b_a_.reshape((-1,1))#b_a_的shape 为（32,1）
        q_next_target=self.target_network(b_s_).detach()
        q_next=q_next_target.gather(1,b_a_)#q_next的shape 为（32,1
        #计算q真实
        q_target=b_r+discount*q_next#q_target的shape 为（32,1）



        loss=self.loss_func(q_predict,q_target)
        # if self.learn_step_counter%5000==0:
        #     print("第",self.learn_step_counter,"次强化学习的loss为：",loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()



#%%

dqn=DQN()
dqn_tradition=DQN()
epoch_cost_optimize=[]
epoch_cost_tradition=[]




epoch_list=[]
loss_list=[]
temp_wu_list=[]
local_costs_epoch=[]
all_costs_epoch=[]
all_costs_epoch_tra=[]
count=0


#每次开始游戏前，重开游戏
#s=env.reset()#s是（4,)的numpy
# print("开始第",epoch+1,"轮游戏")
# start = time.time()
for epoch in range(epochs):
    print("本轮epoch=",epoch,"本轮epoch=",epoch,"本轮epoch=",epoch,"本轮epoch=",epoch)
    step_loss = []
    step_loss_tradition=[]
    local_costs=[]
    all_costs=[]
    all_costs_traditon = []


    #参数随机化
    Hn = np.abs(1.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, [user_num, cap_num]) + np.complex('j') *

                                        np.random.normal(1, 0.1, [user_num, cap_num]))) ** 2
    ##窃听者,[user1,user2]
    Hn_e1 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                           np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e2 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                           np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e3 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                           np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e4 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                           np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e5 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                           np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e6 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                           np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e7 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                           np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e8 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                           np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e9 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                           np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e10 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                           np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e11 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                            np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e12 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                            np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e13 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                            np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e14 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                            np.random.normal(1, 0.1, user_num))) ** 2
    Hn_e15 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                            np.random.normal(1, 0.1, user_num))) ** 2

    # Hn_e = Hn_e1 + Hn_e2 + Hn_e3 + Hn_e4 + Hn_e5 + Hn_e6 + Hn_e7 + Hn_e8 + Hn_e9 + Hn_e10+Hn_e11 + Hn_e12 + Hn_e13 + Hn_e14 + Hn_e15
    # Hn_e=Hn_e1+Hn_e2+Hn_e3+Hn_e4+Hn_e5+Hn_e6+Hn_e7+Hn_e8+Hn_e9+Hn_e10
    Hn_e=Hn_e1
    # Hn_e=Hn_e1

    # Hn_e=Hn_e1
    Hn_0 = Hn[0]
    Hn_1 = Hn[1]

    while((Hn_e > Hn_1).any() or (Hn_e > Hn_0).any()):
        Hn = np.abs(1.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, [user_num, cap_num]) + np.complex('j') *
                                            np.random.normal(1, 0.1, [user_num, cap_num]))) ** 2
        ##窃听者,[user1,user2]

        Hn_e1 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                               np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e2 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                               np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e3 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                               np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e4 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                               np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e5 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                               np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e6 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                               np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e7 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                               np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e8 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                               np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e9 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                               np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e10 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                                np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e11 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                                np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e12 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                                np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e13 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                                np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e14 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                                np.random.normal(1, 0.1, user_num))) ** 2
        Hn_e15 = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, user_num) + np.complex('j') *
                                                np.random.normal(1, 0.1, user_num))) ** 2
        # Hn_e = Hn_e1 + Hn_e2 + Hn_e3 + Hn_e4 + Hn_e5 + Hn_e6 + Hn_e7 + Hn_e8 + Hn_e9 + Hn_e10 + Hn_e11 + Hn_e12 + Hn_e13 + Hn_e14 + Hn_e15
        # Hn_e = Hn_e1 + Hn_e2 + Hn_e3 + Hn_e4 + Hn_e5 + Hn_e6 + Hn_e7 + Hn_e8 + Hn_e9 + Hn_e10
        Hn_e = Hn_e1
        # Hn_e = Hn_e1

        #共谋：sum
        Hn_0 = Hn[0]
        Hn_1 = Hn[1]


    # user_l = np.array([[np.random.randint(10,60), np.random.randint(550,600)]])
    user_l=np.array([[10,550]])
    #--------------------------
    #------------优化功率和算力-------------
    env = environment(user_num=2, cap_num=2, W=5.0,
                      f_local=np.array([[1.4 * 10e8, 0.43 * 10e8]]),
                      omega=330 / 8,
                      F_cap=np.array([[10 * 10e8, 3 * 10e8]]), p_local=1e-1, p_tran=np.array([[2, 3]]), lamuda=1,
                      noise=0.1, Hn_0=Hn_0, Hn_1=Hn_1,Hn_e=Hn_e,user_l=user_l,
                      suspend=sus
                      )
    s = env.reset()
    while True:#每次循环执行一个动作

        #通过神经网络，输入状态，获取动作。(不带学习)
        action=dqn.choose_action(s)

        #执行，获得环境反馈
        s_,r,done,tc_wald,local_cost,all_cost=env.step(action)
        step_loss.append(tc_wald)
        local_costs.append(local_cost)
        all_costs.append(all_cost)
        #将环境的反馈s,a,r,s_传入记忆库。用于经验回放
        dqn.store_transition(s,action,r,s_)

        #神经网络学习（experience replay 经验回放）
        if dqn.memory_count > MEMORY_CAPACITY:
            count+=1
            loss=dqn.learn()


        #判断是否结束
        if done !=0:
            epoch_cost_optimize.append(done)
            local_costs_epoch.append(local_cost)
            all_costs_epoch.append(all_cost)
            break

        #更新状态，进入下一次循环
        s=s_
    #--------------------------------------------


    # ------------传统方法-------------
    env_tradition = environment1(user_num=2, cap_num=2, W=5.0,
                                f_local=np.array([[1.4 * 10e8, 0.43 * 10e8]]),
                                omega=330 / 8,
                                F_cap=np.array([[10 * 10e8, 3 * 10e8]]), p_local=1e-1, p_tran=np.array([[2, 3]]),
                                lamuda=1,
                                noise=0.1, Hn_0=Hn_0, Hn_1=Hn_1, Hn_e=Hn_e, user_l=user_l,
                                suspend=sus
                                )
    s=env_tradition.reset()
    while True:  # 每次循环执行一个动作

        # 通过神经网络，输入状态，获取动作。(不带学习)
        action = dqn_tradition.choose_action(s)

        # 执行，获得环境反馈
        s_, r, done, tc_wald, local_cost,all_cost_traditon = env_tradition.step(action)
        step_loss_tradition.append(tc_wald)
        all_costs_traditon.append(all_cost_traditon)
        # local_costs.append(local_cost)

        # 将环境的反馈s,a,r,s_传入记忆库。用于经验回放
        dqn_tradition.store_transition(s, action, r, s_)

        # 神经网络学习（experience replay 经验回放）
        if dqn_tradition.memory_count > MEMORY_CAPACITY:
            count += 1
            loss = dqn_tradition.learn()

        # 判断是否结束
        if done!=0:
            epoch_cost_tradition.append(done)
            all_costs_epoch_tra.append(all_cost_traditon)
            break

        # 更新状态，进入下一次循环
        s = s_
    # --------------------------------------------


    plt.plot(range(sus), step_loss, color='skyblue', label='optimize_F_P')
    plt.plot(range(sus), step_loss_tradition, color='red', label='tradition')
    plt.plot(range(sus),local_costs,color="yellow",label="local")
    plt.plot(range(sus), all_costs_traditon, color="green", label='all_off_tra')
    plt.legend(loc="best")

    plt.xlabel("step_dqn")
    plt.ylabel("costs")
    plt.show()


# plt.plot(range(epochs),local_costs_epoch, color='yellow', label='locals')
plt.plot(range(epochs),epoch_cost_optimize,color='skyblue', label='optimize_F_P')
plt.plot(range(epochs),epoch_cost_tradition,color='red', label='tradition')
plt.plot(range(epochs),all_costs_epoch,color='purple', label="all_off")
# plt.plot(range(epochs),all_costs_epoch_tra,color='mediumpurple', label="all_off_Tra")

# Writedata.write_to_excel(epoch_cost_optimize,1,"epoch_cost_optimize_E5_3")
# Writedata.write_to_excel(epoch_cost_tradition,1,"epoch_cost_tradition_E5_3")
# Writedata.write_to_excel(local_costs_epoch,1,"local_costs_epoch_E5_3")

plt.xlabel("epoch")
plt.ylabel("costs")
plt.legend(loc="best")
plt.show()

#保存数据

Writedata.write_to_excel(epoch_cost_optimize,1,"./dataset/tra_opi/optimize_cost2.xls")#优化
Writedata.write_to_excel(all_costs_epoch,1,"./dataset/tra_opi/all_costs2.xls")#优化的全卸载
Writedata.write_to_excel(local_costs_epoch,1,"./dataset/tra_opi/local_costs2.xls")#本地
Writedata.write_to_excel(epoch_cost_tradition,1,"./dataset/tra_opi/tradition_costs2.xls")#传统方法