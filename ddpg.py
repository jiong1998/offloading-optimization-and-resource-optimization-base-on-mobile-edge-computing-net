#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# S=[任务量1✖️卸载率1，任务量2✖️卸载率2]
#A=[卸载率1，卸载率2]
import numpy as np
import torch
# from environment1_customized_users_final import environment
from environment_ddpg import environment
from environment_ddpg_tra import environment1
import matplotlib.pyplot as plt

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




n_action,n_state=env.action_state_num()

memory_size=5000#记忆库大小
batch_size=64#batch大小
discount=0.9#折扣率
var1=1#var为方差，用于对a添加高斯分布的噪音
var=1
TAU = 0.01 #用于逐步更新target网络

#%%

#先构建神经网络：

#Actor

class Aactor(torch.nn.Module):
    def __init__(self):
        super(Aactor,self).__init__()
        self.fc1=torch.nn.Linear(n_state,30)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2_user1=torch.nn.Linear(30,15)#user1
        self.fc2_user1.weight.data.normal_(0, 0.1)

        self.fc2_user2=torch.nn.Linear(30,15)##user2
        self.fc2_user2.weight.data.normal_(0, 0.1)

        self.fc3_user1=torch.nn.Linear(15,8)#user1
        self.fc3_user1.weight.data.normal_(0, 0.1)

        self.fc3_user2 = torch.nn.Linear(15, 8)##user2
        self.fc3_user2.weight.data.normal_(0, 0.1)

        self.fc4_user1 = torch.nn.Linear(8, 3)#user1
        self.fc4_user1.weight.data.normal_(0, 0.1)

        self.fc4_user2 = torch.nn.Linear(8, 3)#user2
        self.fc4_user2.weight.data.normal_(0, 0.1)

        self.relu=torch.nn.ReLU()
        self.softmax=torch.nn.Softmax()
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self,s):
        x=self.sigmoid(self.fc1(s))
        x_user1=self.sigmoid(self.fc2_user1(x))
        x_user1 = self.sigmoid(self.fc3_user1(x_user1))
        x_user1=self.fc4_user1(x_user1)
        x_user1 = self.softmax(x_user1)

        x_user2=self.sigmoid(self.fc2_user2(x))
        x_user2 = self.sigmoid(self.fc3_user2(x_user2))
        x_user2 = self.softmax(self.fc4_user2(x_user2))
        x = torch.cat((x_user1, x_user2), dim=1)

        return x

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc_a1=torch.nn.Linear(n_state,16)
        self.fc_a1.weight.data.normal_(0, 0.1)

        self.fc_b1=torch.nn.Linear(n_action,10)
        self.fc_b1.weight.data.normal_(0, 0.1)

        self.fc_ab1=torch.nn.Linear(26,18)#采用简单的加合
        self.fc_ab1.weight.data.normal_(0, 0.1)

        self.fc_ab2=torch.nn.Linear(18,1)
        self.fc_ab2.weight.data.normal_(0, 0.1)

        self.relu=torch.nn.ReLU()
    def forward(self,s,a):#输出s和a，分别输入到一开始不同的网络，然后在第二层的时候加合在一起，在一起训练
        s=self.fc_a1(s)
        a=self.fc_b1(a)
        x=torch.cat((s,a),dim=1)
        x=self.relu(x)
        x=self.relu(self.fc_ab1(x))
        x=self.fc_ab2(x)
        return x

#%%

# criterion=torch.nn.CrossEntropyLoss()
# optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)

class DDPG():
    def __init__(self):
        #构建记忆库
        self.memory=np.zeros((memory_size,n_state*2+n_action+1),dtype=np.float32)
        self.memory_count=0#用于标记存了多少
        #构建Actor网络
        self.Actor_eval=Aactor()
        self.Actor_target=Aactor()
        self.optimizer_Actor=torch.optim.Adam(self.Actor_eval.parameters(),lr=3e-4)

        #构建Critic网络
        self.Critic_eval=Critic()
        self.Critic_target=Critic()
        self.optimizer_Critic=torch.optim.Adam(self.Critic_eval.parameters(),lr=3e-4)
        self.loss_func=torch.nn.MSELoss()

    def choose_action(self,s):#输入状态s，输出动作
        # s=torch.FloatTensor(s).reshape(-1,n_state)
        # a=self.Actor_eval(s).detach()#a为[1,1]
        # a=a.numpy()[0]#a变为array([0.07249027])，因为step（a）需要传入array

        s = torch.FloatTensor(s)
        a=self.Actor_eval(s).detach()
        a=a.numpy()
        return a

        # return a

    def store_transition(self,s,a,r,s_):

        s=s.reshape(-1,)
        a=a.reshape(-1,)
        s_=s_.reshape(-1,)

        transition=np.hstack((s,a,[r],s_))

        #获得索引
        self.index=self.memory_count%memory_size

        #存入memory
        self.memory[self.index,:]=transition
        self.memory_count+=1

        # transition = np.hstack((s, a, [r], s_))
        # index = self.memory_count % memory_size  # replace the old memory with new memory
        # self.memory[index, :] = transition
        # self.memory_count += 1

        if self.memory_count==memory_size:
            print("开始学习！！")

    def learn(self):
        #具体更新，还没看
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        #随机抽样
        sample_index=np.random.choice(memory_size,size=batch_size)#从0-1999中抽batch_size个数,shape为[32,]
        batch_memory=self.memory[sample_index,:]#batch_memory的shape为[32,8]

        b_s = torch.FloatTensor(batch_memory[:, :n_state])#shape为[32,4]
        b_a = torch.FloatTensor(batch_memory[:, n_state:n_state+n_action])#shape为[32,4]
        b_r = torch.FloatTensor(batch_memory[:, n_state+n_action:n_state+n_action+1])#shape为[32,1]
        b_s_ = torch.FloatTensor(batch_memory[:, -n_state:])#shape为[32,3]


        #计算由Actor网络计算出来的S的a值，并打分
        a=self.Actor_eval(b_s)#shape:[32,1]
        q=self.Critic_eval(b_s,a)#shape:[32,1]

        #更新Actor_eval网络
        loss2=-torch.mean(q)
        self.optimizer_Actor.zero_grad()
        loss2.backward()
        self.optimizer_Actor.step()

        #计算St的q_eval
        q_eval=self.Critic_eval(b_s,b_a)
        #计算St+1的q_target
        a_=self.Actor_target(b_s_).detach()#shape:[32,1]
        q_target=self.Critic_target(b_s_,a_).detach()#shape:[32,1]
        q_target=b_r+discount*q_target#shape:[32,1]

        #
        q_eval=q_eval.reshape(-1,1)
        q_target=q_target.reshape(-1,1)

        #更新critic_eval网络
        loss=self.loss_func(q_eval, q_target)
        self.optimizer_Critic.zero_grad()
        loss.backward()
        self.optimizer_Critic.step()


ddpg=DDPG()
ddpg_tradition=DDPG()
epoch_cost_optimize=[]
epoch_cost_tradition=[]

epoch_list=[]
loss_list=[]
temp_wu_list=[]
local_costs_epoch=[]
all_costs_epoch=[]
all_costs_epoch_tra=[]
count=0


#%%

for epoch in range(epochs):
    print("本轮epoch=",epoch,"本轮epoch=",epoch,"本轮epoch=",epoch,"本轮epoch=",epoch)
    step_loss = []
    step_loss_tradition = []
    local_costs = []
    all_costs = []
    all_costs_traditon = []

    # 参数随机化
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
    Hn_e = Hn_e1 + Hn_e2 + Hn_e3 + Hn_e4 + Hn_e5
    # Hn_e=Hn_e1

    # Hn_e=Hn_e1
    Hn_0 = Hn[0]
    Hn_1 = Hn[1]

    while ((Hn_e > Hn_1).any() or (Hn_e > Hn_0).any()):
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
        Hn_e = Hn_e1 + Hn_e2 + Hn_e3 + Hn_e4 + Hn_e5
        # Hn_e = Hn_e1

        # 共谋：sum
        Hn_0 = Hn[0]
        Hn_1 = Hn[1]

    user_l = np.array([[np.random.randint(10, 60), np.random.randint(550, 600)]])
    # ------------优化功率和算力-------------
    env = environment(user_num=2, cap_num=2, W=5.0,
                      f_local=np.array([[1.4 * 10e8, 0.43 * 10e8]]),
                      omega=330 / 8,
                      F_cap=np.array([[10 * 10e8, 3 * 10e8]]), p_local=1e-1, p_tran=np.array([[2, 3]]), lamuda=1,
                      noise=0.1, Hn_0=Hn_0, Hn_1=Hn_1, Hn_e=Hn_e, user_l=user_l,
                      suspend=sus
                      )
    s = env.reset()
    while True:

        #选择动作
        action=ddpg.choose_action(s)

        # 让动作加入高斯分布后输入环境
        # action = np.clip(np.random.normal(action, var1), 0, 1)

        # 执行，获得环境反馈
        s_, r, done, tc_wald, local_cost, all_cost = env.step(action)
        step_loss.append(tc_wald)
        local_costs.append(local_cost)
        all_costs.append(all_cost)

        #存入记忆库
        ddpg.store_transition(s,action,r,s_)

        #学习
        if ddpg.memory_count>memory_size:
            count+=1
            va1r=var1*0.9995 #逐步减少噪音
            ddpg.learn()

        #判断是否结束
        if done:
            epoch_cost_optimize.append(done)
            local_costs_epoch.append(local_cost)
            all_costs_epoch.append(all_cost)
            break

        s=s_
        # ------------传统方法-------------
    env_tradition = environment1(user_num=2, cap_num=2, W=5.0,
                                 f_local=np.array([[1.4 * 10e8, 0.43 * 10e8]]),
                                 omega=330 / 8,
                                 F_cap=np.array([[10 * 10e8, 3 * 10e8]]), p_local=1e-1, p_tran=np.array([[2, 3]]),
                                 lamuda=1,
                                 noise=0.1, Hn_0=Hn_0, Hn_1=Hn_1, Hn_e=Hn_e, user_l=user_l,
                                 suspend=sus
                                 )
    s = env_tradition.reset()
    while True:

        # 选择动作
        action = ddpg_tradition.choose_action(s)

        # 让动作加入高斯分布后输入环境
        # action = np.clip(np.random.normal(action, var), 0, 1)

        # 执行，获得环境反馈
        s_, r, done, tc_wald, local_cost,all_cost_traditon = env_tradition.step(action)
        step_loss_tradition.append(tc_wald)
        all_costs_traditon.append(all_cost_traditon)

        # 存入记忆库
        ddpg_tradition.store_transition(s, action, r, s_)

        # 学习
        if ddpg_tradition.memory_count > memory_size:
            count += 1
            ddpg_tradition.learn()

        # 判断是否结束
        if done:
            epoch_cost_tradition.append(done)
            all_costs_epoch_tra.append(all_cost_traditon)
            break

        s = s_


    plt.plot(range(sus), step_loss, color='skyblue', label='optimize_F_P')
    plt.plot(range(sus), step_loss_tradition, color='red', label='tradition')
    # plt.plot(range(sus),local_costs,color="yellow",label="local")
    # plt.plot(range(sus),all_costs,color="purple",label="all_off")
    # plt.plot(range(sus), all_costs_traditon, color="green", label='all_off_tra')
    plt.legend(loc="best")

    plt.xlabel("step_dqn_15")
    plt.ylabel("costs")
    plt.show()


plt.plot(range(epochs),local_costs_epoch, color='yellow', label='locals')
plt.plot(range(epochs),epoch_cost_optimize,color='skyblue', label='optimize_F_P')
plt.plot(range(epochs),epoch_cost_tradition,color='red', label='tradition')
plt.plot(range(epochs),all_costs_epoch,color='purple', label="all_off")
plt.plot(range(epochs),all_costs_epoch_tra,color='mediumpurple', label="all_off_Tra")

# Writedata.write_to_excel(epoch_cost_optimize,1,"epoch_cost_optimize_E5_3")
# Writedata.write_to_excel(epoch_cost_tradition,1,"epoch_cost_tradition_E5_3")
# Writedata.write_to_excel(local_costs_epoch,1,"local_costs_epoch_E5_3")

plt.xlabel("epoch_e15")
plt.ylabel("costs")
plt.legend(loc="best")
plt.show()
