% 迭代求解当前策略对应的状态值
r_forbidden = -1;
r_boundary = -1;
r_target = 1;
r_blank = 0;
gamma = 0.9;
% 初始化地图
V_table_ref = [...
               -3.8,  -3.8,  -3.6,  -3.1,  -3.2;
               -3.8,  -3.8,  -3.8,  -3.1,  -2.9;
               -3.6,  -3.9,  -3.4,  -3.2,  -2.9;
               -3.9,  -3.6,  -3.4,  -2.9,  -3.2;
               -4.5,  -4.2,  -3.4,  -3.4,  -3.5];
grid = [...
        0 0 0 0 0;
        0 1 1 0 0;
        0 0 1 0 0;
        0 1 0 1 0;
        0 1 0 0 0];
goal = [4,3]

policy = [0.2, 0.2, 0.2, 0.2, 0.2];
% 计算在当前随机策略下的值
% V_table = test_cal_stoa_policy_value();
[V_table, rmse_vec] = cal_stoa_policy_value(grid, policy, r_forbidden, r_boundary, r_target, r_blank, gamma, goal, V_table_ref);
plot(rmse_vec); hold on;
plotVSurfaceLikeSlide(V_table_ref); hold on;
plotVSurfaceLikeSlide(V_table)


function [V_table, rmse_vec] = cal_stoa_policy_value(grid, policy, r_forbidden, r_boundary, r_target, r_blank, gamma, goal, V_table_ref)
row_size = size(grid, 1);
col_size = size(grid, 2);
V_table = zeros(row_size, col_size);
actions = [-1, 0; 0, 1; 1, 0; 0 -1; 0, 0];
% 更新公式位v = sun_{pi}(sum_{r}pi(a|s)p(r|s,a) + gamma*sum_{s'}p(s'|s,a)v(s'))
episodes = 500;
rmse_vec = zeros(episodes, 1);
for ep = 1 : episodes
    for row = 1 : row_size
        for col = 1 : col_size
            temp_val = 0;
            cur_pos = [row, col];
            nex_pos = cur_pos;
            for act = 1 : size(actions, 1)
                cur_action = actions(act, :);
                nex_pos(1) = cur_pos(1) + cur_action(1);
                nex_pos(2) = cur_pos(2) + cur_action(2);
                reward = 0;
                if nex_pos(1) < 1 || nex_pos(1) > row_size || nex_pos(2) < 1 || nex_pos(2) > col_size
                    nex_pos = cur_pos;
                    reward = r_boundary;
                elseif grid(nex_pos(1), nex_pos(2)) == 1
                    reward = r_forbidden;
                elseif isequal(nex_pos, goal)
                    reward = r_target;
                end
                temp_val = temp_val + policy(act)*(reward + gamma*V_table(nex_pos(1), nex_pos(2)));
            end
            V_table(row, col) = temp_val;
        end
    end
    E = V_table - V_table_ref;
    e = E(:);       
    rmse_vec(ep) = sqrt(mean(e.^2));
end

end

function V_table = test_cal_stoa_policy_value()
% 先写一个例子证明策略迭代会收敛到当前pi对应的状态值
grid = [0, 0];
goal = [1, 2];
gamma = 0.9
row_size = size(grid, 1);
col_size = size(grid, 2);
V_table = zeros(row_size, col_size);
actions = [-1, 0; 0, 1; 1, 0; 0 -1; 0, 0];
% 更新公式位v = sun_{pi}(sum_{r}pi(a|s)p(r|s,a) + gamma*sum_{s'}p(s'|s,a)v(s'))
episodes = 500;
for ep = 1 : episodes
    V_table(1, 1) = -1 + gamma*V_table(1, 1);
    V_table(1, 2) = 0 + gamma*V_table(1, 1);
    
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
