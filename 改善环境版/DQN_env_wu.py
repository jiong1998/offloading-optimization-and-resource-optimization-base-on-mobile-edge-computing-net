#%%
import os
import Writedata
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#2021年11。2。 用了裕鑫的环境,自己的模型
# 输入的s为：[任务量1✖️卸载率1，任务量2✖️卸载率2， .....]
# 输出的a为某个用户卸载率增加或者减少

#2cap 2user
import torch
import numpy as np
import Writedata
import matplotlib.pyplot as plt

from Environment_Wu import Environment
from Environment_Wu_tradition import Environment1

#%%

# user_num=input("please input user_num")

#初始化参数-----------------
user_num=2
cap_num=2
users_num=int(user_num)
MAX_EP=2000#步数
epochs=100
e_num=5#窃听数

Hn = np.abs(1.1 * np.sqrt(1 / 2) * (np.random.normal(1, 1, [user_num, cap_num]) + np.complex('j') *
                                        np.random.normal(1, 1, [user_num, cap_num]))) ** 2


Hn_e = np.abs(0.1 * np.sqrt(1 / 2) * (np.random.normal(1, 1, user_num) + np.complex('j') *
                                          np.random.normal(1, 1, user_num))) ** 2##窃听者,[user1,user2]


user_l=np.array([10, 550])

#----------------

env = Environment(user_num=2, cap_num=2, W=5e6,
                  f_local=np.array([1.4 * 10e8, 0.43 * 10e8]),
                  omega=330 / 8,
                  F_cap=np.array([10 * 10e8, 3 * 10e8]), p_local=1e-1, p_tran=np.array([2, 3]), lamda_=1.0,
                  p_noise=0.1, Hn=Hn, Hn_e=Hn_e, user_l=user_l,Mb_to_bit=2 ** 20
                  )


#获取环境的相关参数:
n_action,n_state=env.action_state_num()
#其他超参数的定义：
MEMORY_CAPACITY=5000#记忆库的存储大小
greedy=0.57#贪婪概率
discount=0.9#折扣率       ff
batch_size=32#一次训练32个数据
TARGET_REPLACE_ITER=100#目标网络的更新频率

def get_channel_gain_sum():#共谋
    # Hn (2,2)
    while True:
        Hn = np.abs(1.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, [user_num, cap_num]) + np.complex('j') *
                                            np.random.normal(1, 0.1, [user_num, cap_num]))) ** 2
        Hn_e = np.abs(0.2 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, [user_num, e_num]) + np.complex('j') *
                                                  np.random.normal(1, 0.1, [user_num, e_num]))) ** 2
        Hn_e = np.sum(Hn_e, axis=1)  # 共谋取和
        if(Hn>Hn_e).all():
          break

    return Hn, Hn_e

def get_channel_gain_max():#非共谋
    while True:
        Hn = np.abs(1.1 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, [user_num, cap_num]) + np.complex('j') *
                                            np.random.normal(1, 0.1, [user_num, cap_num]))) ** 2

        Hn_e = np.abs(0.2 * np.sqrt(1 / 2) * (np.random.normal(1, 0.1, [user_num, e_num]) + np.complex('j') *
                                                  np.random.normal(1, 0.1, [user_num, e_num]))) ** 2
        Hn_e=np.max(Hn_e, axis=1)
        if (Hn > Hn_e).all():
            break
    return Hn, Hn_e

#%%

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1=torch.nn.Linear(n_state,64)#(10,20)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2=torch.nn.Linear(64,256)
        self.linear2.weight.data.normal_(0, 0.1)
        self.linear3=torch.nn.Linear(256,64)
        self.linear3.weight.data.normal_(0, 0.1)
        self.linear4=torch.nn.Linear(64,n_action)#(30,12)
        self.linear4.weight.data.normal_(0, 0.1)
        self.relu=torch.nn.ReLU()
        self.dropout=torch.nn.Dropout(p=0.5)
    def forward(self,x):
        x=self.relu(self.linear1(x))
        x=self.dropout(x)
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
        self.agent_network.eval()
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

        self.agent_network.train()

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
r_list_epoch=[]

local_costs_epoch=[]
all_costs_epoch=[]
all_costs_epoch_tra=[]
count=0

#每次开始游戏 前，重开游戏
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
    r_total=0

    Hn, Hn_e = get_channel_gain_sum()
    user_l = np.array([np.random.randint(10,20),np.random.randint(550,560)])

    #--------------------------
    #------------优化功率和算力-------------
    env = Environment(user_num=2, cap_num=2, W=5e6,
                      f_local=np.array([1.4 * 10e8, 0.43 * 10e8]),
                      omega=330 / 8,
                      F_cap=np.array([10 * 10e8, 3 * 10e8]), p_local=2, p_tran=np.array([2, 3]), lamda_=1.0,
                      p_noise=0.1, Hn=Hn, Hn_e=Hn_e, user_l=user_l, Mb_to_bit=2 ** 20
                      )

    s = env.reset()
    episode_step = 0
    while episode_step <MAX_EP:#每次循环执行一个动作
        episode_step+=1
        #通过神经网络，输入状态，获取动作。(不带学习)
        action=dqn.choose_action(s)

        #执行，获得环境反馈
        #return self.state, reward, self.total_local_latency, tc_optimized, self.total_cap_offload_latency, self.done#self.tc_cur为传统方法
        s_,r,local_latency,tc_optimized,all_cost,done=env.step(action,episode_step)
        r_total += r

        step_loss.append(tc_optimized)
        local_costs.append(local_latency)
        all_costs.append(all_cost)
        #将环境的反馈s,a,r,s_传入记忆库。用于经验回放
        dqn.store_transition(s,action,r,s_)

        #神经网络学习（experience replay 经验回放）
        if dqn.memory_count > MEMORY_CAPACITY:
            count+=1
            loss=dqn.learn()


        #判断是否结束
        if done !=0:
            break

        #更新状态，进入下一次循环
        s=s_
    #-------------------------------------------

    print("最后一个step的优化PF的cost=",tc_optimized)

    #------------传统方法-------------
    env_tradition = Environment1(user_num=2, cap_num=2, W=5e6,
                      f_local=np.array([1.4 * 10e8, 0.43 * 10e8]),
                      omega=330 / 8,
                      F_cap=np.array([10 * 10e8, 3 * 10e8]), p_local=2, p_tran=np.array([2, 3]), lamda_=1.0,
                      p_noise=0.1, Hn=Hn, Hn_e=Hn_e, user_l=user_l, Mb_to_bit=2 ** 20
                      )
    s=env_tradition.reset()
    episode_step = 0
    while episode_step < MAX_EP:  # 每次循环执行一个动作
        episode_step += 1
        # 通过神经网络，输入状态，获取动作。(不带学习)
        action = dqn_tradition.choose_action(s)

        # 执行，获得环境反馈
        #return self.state, reward, self.tc_cur, self.done
        s_, r,  tc_tradition,  done = env_tradition.step(action, episode_step)
        step_loss_tradition.append(tc_tradition)
        # 将环境的反馈s,a,r,s_传入记忆库。用于经验回放
        dqn_tradition.store_transition(s, action, r, s_)

        # 神经网络学习（experience replay 经验回放）
        if dqn_tradition.memory_count > MEMORY_CAPACITY:
            count += 1
            loss = dqn_tradition.learn()

        # 判断是否结束
        if done != 0:
            break

        # 更新状态，进入下一次循环
        s = s_
    #--------------------------------------------
    epoch_cost_optimize.append(np.mean(step_loss[-200:]))  # 强化学习、优化PF、cost
    local_costs_epoch.append(local_latency)  # 本地
    all_costs_epoch.append(all_cost)  # 优化的all cap
    epoch_cost_tradition.append(np.mean(step_loss_tradition[-200:])) # 强化学习、均分PF、cost
    r_list_epoch.append(r_total)

    plt.plot(range(episode_step), step_loss, color='skyblue', label='optimize_F_P')
    plt.plot(range(episode_step), step_loss_tradition, color='red', label='tradition')
    plt.plot(range(episode_step),local_costs,color="yellow",label="local")
    plt.plot(range(episode_step),all_costs,color="purple",label="all_off")
    # plt.plot(range(episode_step), all_costs_traditon, color="green", label='all_off_tra')
    plt.legend(loc="best")

    plt.xlabel("step_dqn_e1")
    plt.ylabel("costs")
    plt.show()
    if epoch <20:
        if greedy<0.73:
            if epoch %5 ==0:
                greedy+=0.03
    if epoch==30:
        greedy=0.85
    if epoch >30 and epoch <=40:
        if epoch %5 == 0:
            greedy+=0.01
    if epoch > 40:
        if epoch %10 ==0:
            if greedy<0.98:
                greedy+=0.02

    if epoch==98:
        greedy=0.99
    print("greedy=",greedy)

plt.plot(range(epochs),local_costs_epoch, color='yellow', label='locals')
plt.plot(range(epochs),epoch_cost_optimize,color='skyblue', label='optimize_F_P')
plt.plot(range(epochs),epoch_cost_tradition,color='red', label='tradition')
plt.plot(range(epochs),all_costs_epoch,color='purple', label="all_off")
# plt.plot(range(epochs),all_costs_epoch_tra,color='mediumpurple', label="all_off_Tra")

plt.xlabel("epoch_colluding")
plt.ylabel("costs")
plt.legend(loc="best")
plt.show()

#保存数据

Writedata.write_to_excel(epoch_cost_optimize,1,"./dataset/tra_opi/optimize_cost2.xls")#优化
Writedata.write_to_excel(all_costs_epoch,1,"./dataset/tra_opi/all_costs2.xls")#优化的全卸载
Writedata.write_to_excel(local_costs_epoch,1,"./dataset/tra_opi/local_costs2.xls")#本地
Writedata.write_to_excel(epoch_cost_tradition,1,"./dataset/tra_opi/tradition_costs2.xls")#传统方法
Writedata.write_to_excel(r_list_epoch,1,"./dataset/tra_opi/r_list.xls")#传统方法
