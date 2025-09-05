import numpy as np
import torch
from typing import Tuple, List
from dataclasses import dataclass
import matplotlib.pylab as plt
import random
import torch.nn as nn
import torch.optim as optim

@dataclass
class GridConfig:
    grid: np.ndarray
    start_rc: Tuple
    goal_rc: Tuple
    r_forbidden: float = -0.5
    r_boundary: float = -0.3
    r_step: float = -0.01
    r_target: float = 1.0
    max_steps_for_eps: int = 500
    gamma: float = 0.95

class GridWorld:
    def __init__(self, cfg: GridConfig):
        self.cfg = cfg
        self.H, self.W = cfg.grid.shape
        self.nS = self.H* self.W
        self.nA = 4
        self.actions: List[Tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.start_s = self.rc2i(cfg.start_rc[0], cfg.start_rc[1])
        self.goal_s = self.rc2i(cfg.goal_rc[0], cfg.goal_rc[1])

    def rc2i(self, r, c):
        return r* self.W + c
    
    def i2rc(self, ind: int):
        r = ind // self.W
        c = ind % self.W
        return (r, c)

    # 生成环境的任意非禁止状态
    def reset(self):
        # 返回任意为0的位置的索引
        free = np.argwhere(self.cfg.grid == 0)
        # 去除目标点
        choices = [(r, c) for r, c in free if (r, c) != self.cfg.goal_rc]
        # 从剩余位置随机选择一个
        self.r, self.c = random.choice(choices)
        self.t = 0
        return self.state()
     
    def state(self):
        return self.rc2i(self.r, self.c)
    
    def env_step(self, state: int, act):
        r, c = self.i2rc(state)
        action = self.actions[act]
        nr = r + action[0]
        nc = c + action[1]
        reward = 0
        done = False
        # 判断nr, nc状态
        if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
            reward = self.cfg.r_boundary
        elif self.cfg.grid[nr][nc] == 1:
            reward = self.cfg.r_forbidden
        elif (nr, nc) == self.cfg.goal_rc:
            reward = self.cfg.r_target
            state = self.rc2i(nr, nc)
            done = True
        else:
            reward = self.cfg.r_step
            state = self.rc2i(nr, nc)
        return state, reward, done

class MathToolForREINFORCE:
    def __init__(self):
        pass

    def softmax_func(self, x):
        # x: (B, nA) 或 (nA,)
        if x.dim() == 1:
            z = x - x.max()
            e = torch.exp(z)
            return e / e.sum()
        else:
            z = x - x.max(dim=-1, keepdim=True).values
            e = torch.exp(z)
            return e / e.sum(dim=-1, keepdim=True)

    def sample_from(self, p):
        # num_samples = 1是返回一次采样的结果
        idx = torch.multinomial(p, num_samples=1, replacement=True).item()
        return idx
    
    def greedy_action(self, pi_s):
        return int(np.argmax(pi_s))
    
    def one_hot(self, cur_s, nS):
        x = np.zeros(nS, dtype=np.float32)
        x[cur_s] = 1.0
        return x

# REINFORCE是on-policy的，不可以使用replay
class PolicyGradientNet(nn.Module):
    def __init__(self, nS: int, nA: int, hidden = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, nA)
        )

    def forward(self, x):
        out = self.model(x)
        return out

def train_REINFORCE(env: GridWorld, episodes=5000, alpha=0.01,
                    use_baseline=False, beta=0.2, normalize_adv=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("Using device:", device)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    # 初始化策略pi的参数theta
    tool = MathToolForREINFORCE()
    reward_hist = np.zeros(episodes)
    V = np.zeros(env.nS, dtype=np.float64)
    policy_net = PolicyGradientNet(env.nS, env.nA).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    policy_net.train()
    for eps in range(episodes):
        # 采样一条轨迹
        cur_s = env.reset()
        S, A, R = [], [], []
        logprob = []
        for step in range(env.cfg.max_steps_for_eps):
            if cur_s == env.goal_s:
                break
            # 将状态输入到神经网络，输出对应状态下的策略
            input = torch.tensor(tool.one_hot(cur_s, env.nS), dtype=torch.float32, device=device).unsqueeze(0)
            # 使用torch自带的函数，保证可导，dim=-1按照最后一维进行运算，在这里面就是按照行进行softmax运算
            pi_s = torch.softmax(policy_net(input), dim=-1)
            pi_s = pi_s.squeeze()
            # 使用Categorical分布实现可导采样，创建分布
            dist = torch.distributions.Categorical(pi_s)
            # 分布采样得到索引，可导
            act = dist.sample()
            # print(act.grad)
            log_prob = dist.log_prob(act)
            logprob.append(log_prob)
            # print(log_prob.grad)
            state, reward, done = env.env_step(cur_s, act.item())
            # 使用自定义函数进行softmax 采样
            # pi_s = policy_net(input)
            # pi_s = tool.softmax_func(pi_s)
            # # 采样动作索引
            # act = tool.sample_from(pi_s)
            # state, reward, done = env.env_step(cur_s, act)
            # logprob.append(torch.log(pi_s[0, act]))
            S.append(cur_s)
            A.append(act)
            R.append(reward)
            cur_s = state
            if done is True:
                break
        # 计算MC采样回报
        length = len(R)
        G = np.zeros(length, dtype=np.float32)
        g = 0
        for t in range(length-1, -1, -1):
            g = R[t] + env.cfg.gamma* g
            G[t] = g
        # G = (G - G.mean()) / (G.std() + 1e-8)
        reward_hist[eps] = np.sum(R)
        logprob_tensor = torch.stack(logprob)
        g_tensor = torch.tensor(G, dtype=torch.float32, device=device)
        # 进行梯度更新,定义损失函数
        if length > 0:
            policy_loss = -(logprob_tensor * g_tensor).sum()
            # policy_loss = -(logprob_tensor * g_tensor).mean()
            optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
            optimizer.step()
        if (eps+1) % 10 == 0:
            print(f"Episode {eps+1:4d} | Return = {reward_hist[eps]:.3f}")
    return policy_net, V, reward_hist

def draw_episode(env: GridWorld, policy_net, tool: MathToolForREINFORCE):
    policy_net.eval()
    H, W = env.cfg.grid.shape
    bg = env.cfg.grid.astype(float).copy()
    sr, sc = env.cfg.start_rc
    gr, gc = env.cfg.goal_rc
    plt.figure(figsize=(8, 6))
    plt.imshow(bg, cmap='gray_r')
    plt.xticks(np.arange(-0.5, W, 1))
    plt.yticks(np.arange(-0.5, H, 1))
    plt.grid(color='k', linestyle='-', linewidth=0.6)
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    U = np.zeros_like(bg); Vv = np.zeros_like(bg)
    for r in range(H):
        for c in range(W):
            if env.cfg.grid[r][c] == 1:
                continue
            state = env.rc2i(r, c)
            input = torch.tensor(tool.one_hot(state, env.nS), dtype=torch.float32).unsqueeze(0)
            pi_s = policy_net(input)
            pi_s = tool.softmax_func(pi_s)
            a_ind = tool.greedy_action(pi_s.detach().numpy())
            if a_ind == 0: 
                U[r,c]=0
                Vv[r,c]=-0.4
            if a_ind == 1: 
                U[r,c]=0.4
                Vv[r,c]=0
            if a_ind == 2: 
                U[r,c]=0   
                Vv[r,c]=0.4
            if a_ind == 3: 
                U[r,c]=-0.4
                Vv[r,c]=0
    plt.quiver(X, Y, U, Vv, angles='xy', scale_units='xy', scale=1, color='k')
    plt.scatter([sc],[sr], s=80, c='g', label='start')
    plt.scatter([gc],[gr], s=80, c='b', label='goal')
    plt.legend(loc='upper right')
    plt.title("Final greedy policy")

if __name__ == "__main__":
    # grid = np.array([
    #     [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,0,1,1,0],
    #     [1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0],
    #     [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1,0],
    #     [0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0],
    #     [0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,1,1,0,0,0,0,0,1,0,0],
    #     [1,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0],
    #     [0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
    #     [0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,1,0,1,1,1,0,0,1,1,0,0,0],
    #     [1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1,1,1,1,0],
    #     [1,0,1,0,0,1,0,1,0,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0],
    #     [1,0,1,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0],
    #     [0,0,0,0,1,1,0,0,0,1,0,0,1,0,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0],
    #     [0,1,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1],
    #     [1,1,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,0,0,1,1,1,0,0,0,0,1,1,0,0],
    #     [0,1,0,0,1,0,0,0,0,0,1,1,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1],
    #     [1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,1,1,0,1,0,1,1,0,0,0,1,0,1],
    #     [1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0],
    #     [0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0],
    #     [1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0],
    #     [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0]
    # ], dtype=int)
    grid = np.array([[0,0,0,0,0], [0,1,1,0,0], [0,0,1,0,0], [0,1,0,1,0], [0,1,0,0,0]], dtype=int)
    start0 = (0, 0)
    goal0  = (3, 2)
    cfg = GridConfig(grid, start0, goal0)
    env = GridWorld(cfg)
    tool = MathToolForREINFORCE()
    policy_net, V, reward_hist = train_REINFORCE(env)
    draw_episode(env, policy_net, tool)
    xx = np.arange(0, len(reward_hist), 1)
    plt.figure(figsize=(8, 6))
    plt.plot(xx, reward_hist)
    plt.grid()
    plt.show()
    











    





        


    

