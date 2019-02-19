%% (mini-batch) Stochastic variance reduced gradient (SVRG)
function [w_vs_iter, cost_vs_iter, step_vs_iter, norm_grad_vs_iter] = stochastic_variance_reduced_gradient(X, y, N, algo_struct)

w_init                 = algo_struct.w_init;
cost_func_handle       = algo_struct.cost_func_handle;
grad_handle            = algo_struct.grad_handle;
grad_per_sample_handle = algo_struct.grad_per_sample_handle;
nrof_iter              = algo_struct.nrof_iter;
nrof_iter_inner_loop   = algo_struct.nrof_iter_inner_loop; % EPOCH LENGTH
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

cost_vs_iter         = ones(nrof_iter+1, 1); % +1 for initialization
cost_vs_iter(1)      = cost_func_handle(X, y, N, w_init, lambda);

norm_grad_vs_iter = zeros(nrof_iter+1, 1);

w_svrg               = w_init;
step_alpha           = step_size; % initial
counter              = 0;
for kk_outer = 1:nrof_iter % outer-loop /nr of epochs
    
       
    %full_grad     = grad_handle(X, y, N, w_svrg, lambda); % compute full gradient
    full_grad_per_sample  = grad_per_sample_handle(X, y, N, w_svrg, lambda); % compute full gradient
    full_grad             = mean(full_grad_per_sample, 2);
    
    w_prev_svrg           = w_svrg; % store w
    
    cost_vs_iter(kk_outer+1) = cost_func_handle(X, y, N, w_svrg, lambda);
        
    for tt_inner = 1:nrof_iter_inner_loop
        
        [perm_indices]     = randperm(N,mini_batch_size);
        X_mini_batch       = X(perm_indices,:);
        y_mini_batch       = y(perm_indices);
        
        counter            = counter + 1;
        switch lower(step_size_method)
            case 'fixed'
                step_alpha = step_size;
            case {'decay'; 'decay1'}
                step_alpha = step_size / (1 + tt_inner);
            case 'adaptive' % different from other methods
                step_alpha = step_size / (1 + step_size * lambda * counter);
            case 'adaptive_bb'
                if tt_inner > 1
                    delta_grad  = grad_w_current_svrg - grad_w_prev_svrg;
                    delta_w     = w_svrg - w_prev_svrg;
                    step_alpha = compute_step_size__barzilai_borwein_method(delta_w, delta_grad);
                    if isnan(step_alpha) || (step_alpha==inf) || (step_alpha<=0)
                        step_alpha = step_size; % better be safe
                    end
                else
                    step_alpha = step_size;
                end
            otherwise
                error('unknown step size computation method');
        end
        
        % calculate gradient
        %grad_w_prev_svrg     = grad_handle(X_mini_batch, y_mini_batch, mini_batch_size, w_prev_svrg, lambda);
        %grad_w_current_svrg  = grad_handle(X_mini_batch, y_mini_batch, mini_batch_size, w_svrg, lambda);   
        
        grad_w_prev_svrg_per_sample     = grad_per_sample_handle(X_mini_batch, y_mini_batch, mini_batch_size, w_prev_svrg, lambda);
        grad_w_current_svrg_per_sample  = grad_per_sample_handle(X_mini_batch, y_mini_batch, mini_batch_size, w_svrg, lambda);        
        grad_w_prev_svrg                = mean(grad_w_prev_svrg_per_sample, 2);
        grad_w_current_svrg             = mean(grad_w_current_svrg_per_sample, 2);
        
        % update weights
        v      = full_grad + grad_w_current_svrg - grad_w_prev_svrg;
        w_svrg = w_svrg - step_alpha * v;
        
        w_vs_iter(:,kk_outer+1)  = w_svrg;
        cost_vs_iter(kk_outer+1) = cost_func_handle(X, y, N, w_svrg, lambda);
        
        step_vs_iter(kk_outer+1) = step_alpha;
        
        norm_grad_vs_iter(kk_outer) = norm(grad_w_prev_svrg);
        norm_grad_vs_iter(kk_outer+1) = norm(grad_w_current_svrg);
        
    end
end

% Saving data
% (mini-batch) SVRG
str_mbSVRG = strcat('CA2_results/mbSVRG_',algo_struct.alpha_str,'_Lambda',num2str(lambda),...
            '_BatchS',num2str(mini_batch_size),'_Epoch',num2str(nrof_iter_inner_loop));
save(strcat(str_mbSVRG,'.mat'),'w_vs_iter','cost_vs_iter','step_vs_iter',...
    'norm_grad_vs_iter');

end