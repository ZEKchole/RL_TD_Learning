# REINFORCE
import numpy as np
import torch
from typing import Tuple, List
from dataclasses import dataclass
import matplotlib.pylab as plt
import random

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
        # 平移防止指数过大溢出
        x = x - np.max(x)
        e = np.exp(x)
        return e/ np.sum(e)

    def sample_from(self, p):
        ind = np.random.choice(len(p), p=p)
        return ind
    
    def greedy_action(self, pi_s):
        return int(np.argmax(pi_s))

def train_REINFORCE(env: GridWorld, episodes=5000, alpha=0.01,
                    use_baseline=False, beta=0.2, normalize_adv=True):
    # 初始化策略pi的参数theta
    tool = MathToolForREINFORCE()
    theta = np.zeros((env.nS, env.nA), dtype=np.float32)
    V = np.zeros(env.nS, dtype=np.float64)
    reward_hist = np.zeros(episodes)
    np.random.seed(0)
    for eps in range(episodes):
        # 采样一条轨迹
        cur_s = env.reset()
        S, A, R = [], [], []
        for step in range(env.cfg.max_steps_for_eps):
            if cur_s == env.goal_s:
                break
            # 取出状态s对应的策略，归一化
            pi_s = tool.softmax_func(theta[cur_s, :])
            pi_s = np.array(pi_s, dtype=np.float32)
            # 采样动作索引
            act = tool.sample_from(pi_s)
            state, reward, done = env.env_step(cur_s, act)
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
        reward_hist[eps] = np.sum(R)
        # 进行梯度更新
        if length > 0:
            if use_baseline is True:
                # 带基线的处理,加入基准线可以更好的去评估动作的好坏，不然的话获得低回报的动作也会在算法作用下概率上升
                # 同时加入基准线(和动作无关)不会影响梯度期望值
                A_adv = G - V[np.array(S)]
                # 优势标准化，进行标准化是因为不同回合的A_adv可能相差巨大
                # 进行标准化可以统一梯度的量级，更好的训练
                if normalize_adv:
                    m, s = A_adv.mean(), A_adv.std()
                    if s < 1e-8: s = 1.0
                    A_adv = (A_adv - m) / s
            else:
                A_adv = G.copy()
            for t in range(length):
                s_t, a_t = S[t], A[t]
                # 策略梯度为one_hot(1) - pi_s
                pi_s = tool.softmax_func(theta[s_t])
                grad = -pi_s
                grad[a_t] = grad[a_t] + 1
                theta[s_t] = theta[s_t] + alpha* A_adv[t]* grad 
                # 更新基线
                if use_baseline:
                    V[s_t] += beta * (G[t] - V[s_t])
        # 打印reward
        if (eps+1) % 200 == 0:
            print(f"Episode {eps+1:4d} | Return = {reward_hist[eps]:.3f}")
    return theta, V, reward_hist

def draw_episode(env: GridWorld, theta, tool: MathToolForREINFORCE):
    # 先绘制单元格
    H, W = env.cfg.grid.shape
    bg = env.cfg.grid.astype(float).copy()
    sr, sc = env.cfg.start_rc
    gr, gc = env.cfg.goal_rc
    # 绘制栅格地图
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='gray_r')
    plt.xticks(np.arange(-0.5, W, 1))
    plt.yticks(np.arange(-0.5, H, 1))
    plt.grid(color='k', linestyle='-', linewidth=0.6)
    # 最优策略得到path
    path_x = []
    path_y = []
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    U = np.zeros_like(bg); Vv = np.zeros_like(bg)
    for r in range(H):
        for c in range(W):
            if env.cfg.grid[r][c] == 1:
                continue
            state = env.rc2i(r, c)
            a_ind = tool.greedy_action(tool.softmax_func(theta[state]))
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
    theta, V, reward_hist = train_REINFORCE(env)
    draw_episode(env, theta, tool)
    xx = np.arange(0, len(reward_hist), 1)
    plt.figure(figsize=(8, 6))
    plt.plot(xx, reward_hist)
    plt.grid()
    plt.show()
    











    





        


    

