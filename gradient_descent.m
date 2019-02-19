%% Full Gradient Descent
function [ cost_vs_iter, step_vs_iter,norm_grad1_vs_iter] = gradient_descent(X, y, N, algo_struct)

%w_init                 = algo_struct.w_init;
w3_init                 = algo_struct.w3_init;
W2_init                 = algo_struct.W2_init;
W1_init                 = algo_struct.W1_init;
cost_func_handle        = algo_struct.cost_func_handle;
%grad_handle            = algo_struct.grad_handle;
grad_J_cost_L2_w3       = algo_struct.grad_w3_handle;
grad_J_cost_L2_W2       = algo_struct.grad_W2_handle;
grad_J_cost_L2_W1       = algo_struct.grad_W1_handle;
%grad_per_sample_handle = algo_struct.grad_per_sample_handle;
nrof_iter              = algo_struct.nrof_iter;
step_size              = algo_struct.step_size; % fixed value is used
%step_size_method       = algo_struct.step_size_method;
%lambda                 = algo_struct.lambda_reg;
%step_size_handle  = algo_struct.step_size_handle;

% w3_vs_iter         = zeros(numel(w3_init), nrof_iter+1);
% w3_vs_iter(:,1)    = w_init;

step_vs_iter      = zeros(nrof_iter+1, 1);
step_vs_iter(1)   = step_size;

norm_grad3_vs_iter = zeros(nrof_iter+1,1);
norm_grad2_vs_iter = zeros(nrof_iter+1,1);
norm_grad1_vs_iter = zeros(nrof_iter+1,1);


cost_vs_iter      = ones(nrof_iter+1, 1); % +1 for initialization
cost_vs_iter(1)   = cost_func_handle(X.', y.', N, W1_init, W2_init, w3_init);

%w_gd              = w_init;

w3_gd    = w3_init;              
W2_gd    = W2_init;
W1_gd    = W1_init;

d = size(X,2);

for kk_outer = 1:nrof_iter
    
    step_alpha = step_size;
    
    grad_w3 = zeros(d,1);
    grad_W2 = zeros(d,d);
    grad_W1 = zeros(d,d);

    
    for kk_N = 1:N
        grad_w3 = grad_w3 + grad_J_cost_L2_w3(X(kk_N,:).', y(kk_N),N, W1_gd , W2_gd,w3_gd);
        grad_W2 = grad_W2 + grad_J_cost_L2_W2(X(kk_N,:).', y(kk_N),N, W1_gd , W2_gd,w3_gd);
        grad_W1 = grad_W1 + grad_J_cost_L2_W1(X(kk_N,:).', y(kk_N),N, W1_gd , W2_gd,w3_gd);
    end

    
    %w_gd_prev                 = w_gd;
    %grad_w_prev              = grad_handle(X, y, N, w_gd_prev, lambda);
%     grad_w_prev_per_sample    = grad_per_sample_handle(X, y, N, w3_gd, );
%     grad_w_prev               = mean(grad_w_prev_per_sample, 2);
    
    % Update Weights
    w3_gd = w3_gd - step_alpha*grad_w3;
    W2_gd = W2_gd - step_alpha*grad_W2;
    W1_gd = W1_gd - step_alpha*grad_W1;
    
    %w_gd                      = w_gd - step_alpha* grad_w_prev;
    
    %grad_w_current           = grad_handle(X, y, N, w_gd, lambda);
%     grad_w_current_per_sample = grad_per_sample_handle(X, y, N, w_gd, lambda);
%     grad_w_current            = mean(grad_w_current_per_sample,2);
    
    %w_vs_iter(:,kk_outer+1)   = w_gd;
    cost_vs_iter(kk_outer+1)  = cost_func_handle(X.', y.', N, W1_gd , W2_gd,w3_gd);
    
    step_vs_iter(kk_outer+1)  = step_alpha;
    
    %norm_grad_vs_iter(kk_outer) = norm(grad_w_prev);
    norm_grad3_vs_iter(kk_outer+1) = norm(grad_w3,2);
    norm_grad2_vs_iter(kk_outer+1) = norm(grad_W2,2);
    norm_grad1_vs_iter(kk_outer+1) = norm(grad_W1,2);    
end

% Saving data
% Gradient Descent
str_GD = strcat('CA3_results/fullGD_',algo_struct.alpha_str);
save(strcat(str_GD,'.mat'),'cost_vs_iter','step_vs_iter',...
    'norm_grad1_vs_iter','norm_grad2_vs_iter','norm_grad3_vs_iter');

end