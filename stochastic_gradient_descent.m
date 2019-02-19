%% (mini-batch) Stochastic Gradient Descent
function [w_vs_iter, cost_vs_iter, step_vs_iter, norm_grad_vs_iter] = stochastic_gradient_descent(X, y, N, algo_struct)

w_init                 = algo_struct.w_init;
cost_func_handle       = algo_struct.cost_func_handle;
grad_handle            = algo_struct.grad_handle;
grad_per_sample_handle = algo_struct.grad_per_sample_handle;
nrof_iter              = algo_struct.nrof_iter;
step_size              = algo_struct.step_size; % fixed value is used
step_size_method       = algo_struct.step_size_method;
mini_batch_size        = algo_struct.mini_batch_size; % mini_batch_size==1 => SGD ... mini_batch_size > 1 => mini_batch SGD
mini_batch_rng_gen     = algo_struct.mini_batch_rng_gen; % random number
lambda                 = algo_struct.lambda_reg;

rng(mini_batch_rng_gen);

w_vs_iter            = zeros(numel(w_init), nrof_iter+1);
w_vs_iter(:,1)       = w_init;

step_vs_iter         = zeros(nrof_iter+1, 1);
step_vs_iter(1)      = step_size;

norm_grad_vs_iter = zeros(nrof_iter+1, 1);

cost_vs_iter         = ones(nrof_iter+1, 1); % +1 for initialization
cost_vs_iter(1)      = cost_func_handle(X, y, N, w_init, lambda);
%step_size_handle  = algo_struct.step_size_handle;


w_sgd = w_init;
for kk_outer = 1:nrof_iter
    [perm_indices]     = randperm(N,mini_batch_size);
    X_mini_batch       = X(perm_indices,:);
    y_mini_batch       = y(perm_indices);
    
    switch lower(step_size_method)
        case 'fixed'
            step_alpha = step_size;
        case {'decay'; 'decay1'}
            step_alpha = step_size / (1 + kk_outer);
        case 'adaptive' % different from other methods
            step_alpha = step_size / (1 + step_size * lambda * kk_outer);
        case 'adaptive_bb'
            if kk_outer > 1
                delta_grad  = grad_w_current - grad_w_prev;
                delta_w     = w_sgd - w_sgd_prev;
                step_alpha  = compute_step_size__barzilai_borwein_method(delta_w, delta_grad);
                if isnan(step_alpha) || (step_alpha==inf) || (step_alpha<=0)
                    step_alpha = step_size; % better be safe
                end
            else
                step_alpha = step_size;
            end
            
        otherwise
            error('unknown step size computation method');
    end
    
    w_sgd_prev                = w_sgd;    
    %grad_w_prev              = grad_handle(X_mini_batch, y_mini_batch, mini_batch_size, w_sgd_prev, lambda);   
    grad_w_prev_per_sample    = grad_per_sample_handle(X_mini_batch, y_mini_batch, mini_batch_size, w_sgd_prev, lambda);  
    grad_w_prev               = mean(grad_w_prev_per_sample, 2);
    
    % update weights
    w_sgd                     = w_sgd - step_alpha* grad_w_prev;
    
    %grad_w_current           = grad_handle(X_mini_batch, y_mini_batch, mini_batch_size, w_sgd, lambda);
    grad_w_current_per_sample = grad_per_sample_handle(X_mini_batch, y_mini_batch, mini_batch_size, w_sgd, lambda);
    grad_w_current            = mean(grad_w_current_per_sample,2);
    
    w_vs_iter(:,kk_outer+1)   = w_sgd;
    cost_vs_iter(kk_outer+1)  = cost_func_handle(X, y, N, w_sgd, lambda);
    
    step_vs_iter(kk_outer+1)  = step_alpha;
        
    norm_grad_vs_iter(kk_outer) = norm(grad_w_prev);
    norm_grad_vs_iter(kk_outer+1) = norm(grad_w_current);
end

% Saving data
% (mini-batch) SGD
str_mbSGD = strcat('CA2_results/mbSGD_',algo_struct.alpha_str,'_Lambda',num2str(lambda),...
            '_BatchS',num2str(mini_batch_size));
save(strcat(str_mbSGD,'.mat'),'w_vs_iter','cost_vs_iter','step_vs_iter',...
    'norm_grad_vs_iter');

end