V_table_ref = [...
   -3.8,  -3.8,  -3.6,  -3.1,  -3.2;   % ← 顶行里除了第4列，其它数值请以原图为准修改
   -3.8,  -3.8,  -3.8,  -3.1,  -2.9;
   -3.6,  -3.9,  -3.4,  -3.2,  -2.9;
   -3.9,  -3.6,  -3.4,  -2.9,  -3.2;
   -4.5,  -4.2,  -3.4,  -3.4,  -3.5];
% plotVSurfaceLikeSlide(V)

% 用TD-Linear算法去近似V_table_ref
r_frobidden = -10;
r_boundary = -10;
r_target = 1;
r_blank = 0;
gamma = 0.9;
grid = [...
        0 0 0 0 0;
        0 1 1 0 0;
        0 0 1 0 0;
        0 1 0 1 0;
        0 1 0 0 0];
goal = [4, 3];
policy = ones(1, 5)*0.2;
actions = [-1, 0; 0, 1; 1, 0; 0 -1; 0, 0];
alpha = 0.05;
gamma = 0.9;
[V_table, w, rmse_vec] = TD_Linear_Main2(grid, goal, alpha, policy, actions, gamma, V_table_ref);
plotVSurfaceLikeSlide(V_table)
figure(3)
plot(rmse_vec)

% % 上面的更新将整个网格都扫描了一遍，这样不太好，现在修改为采样一条500steps的episode
function [V_table, w, rmse_vec] = TD_Linear_Main2(grid, goal, alpha0, policy, actions, gamma, V_table_ref)
    row_size = size(grid, 1);
    col_size = size(grid, 2);
    V_table = zeros(row_size, col_size);
    tau = 400;
    % 初始化参数w0
    w = zeros(3, 1);
    w_bar = zeros(3, 1);
    episodes = 500;
    steps = 500;
    t_global = 0;
    rmse_vec = zeros(episodes, 1);
    for eps = 1 : episodes
        cur_pos = [randi(row_size), randi(col_size)];
        for step = 1 : steps
            t_global = t_global + 1;
            alpha = alpha0 / (1 + t_global / tau);   % Robbins–Monro 衰减
            % 随机选取动作，按照概率进行采样
            action_index = randsample(1:5, 1, true, policy);
            % 更新状态
            nex_pos = cur_pos + actions(action_index, :);
            reward = 0;
            if nex_pos(1) < 1 || nex_pos(1) > row_size || nex_pos(2) < 1 || nex_pos(2) > col_size
                reward = -1;
                nex_pos = cur_pos;
            elseif grid(nex_pos(1), nex_pos(2)) == 1
                reward = -1;
            elseif isequal(nex_pos, goal)
                reward = 1;
            end
            % 特征归一化
            x = (cur_pos(2) - 1)/(col_size - 1);
            y = (cur_pos(1) - 1)/(row_size - 1);
            nex = (nex_pos(2) - 1)/(col_size - 1);
            ney = (nex_pos(1) - 1)/(row_size - 1);
            phi_s = [1; x; y];
            phi_s_nex = [1; nex; ney];
            w = w + alpha*(reward + gamma*phi_s_nex'*w - phi_s'*w)*phi_s;
            cur_pos = nex_pos;
        end
        % 一轮w更新完毕
        w_bar = ((eps-1)/eps) * w_bar + (1/eps) * w;
        for row = 1 : row_size
            for col = 1 : col_size
                % 特征归一化
                x = (col - 1)/(col_size - 1);
                y = (row - 1)/(row_size - 1);
                V_table(row, col) = [1; x; y]'*w_bar;
            end
        end
        E = V_table - V_table_ref;
        e = E(:);       
        rmse_vec(eps) = sqrt(mean(e.^2));
    end
end

function plotVSurfaceLikeSlide(V)
% V: HxW 的状态值表（第1维=行 row，第2维=列 column）
    [H, W] = size(V);

    % 用 ndgrid 得到“行=横轴、列=纵轴”的网格（更贴近矩阵索引语义）
    [R, C] = ndgrid(1:H, 1:W);   % R=row (X 轴), C=column (Y 轴)

    figure('Color','w'); hold on;

    % 先画彩色表面（平滑着色），再叠加网格线
    surf(R, C, V, ...
        'FaceColor','interp', ...        % 平滑着色
        'EdgeColor',[0.4 0.4 0.4], ...   % 网格线颜色
        'FaceAlpha',0.98);
    % 也可以单独叠加一层网格：mesh(R, C, V, 'EdgeColor',[0.3 0.3 0.3], 'FaceColor','none');

    colormap(parula);    % 或 turbo
    colorbar;
    grid on; axis tight;

    % 轴刻度与标签（和截图一样）
    set(gca, 'XTick', 1:H, 'YTick', 1:W);
    xlabel('row'); ylabel('column'); zlabel('V(s)');
    title('True state value');

    % 视角调成类似截图效果
    view(45, 30);   % 可微调，如 view(135, 25)
end
