function main_ca3_try()
% (MLoNs) Computer Assignment - 3
% Group 3

%% 
clear variables;

close all;

clc;

rng(0); 

%% Load data
% Percentage of data for training
prcntof_data_for_training = 0.8;
% Load household (1) or crimes (0) dataset
flagData = 0;
% 1 means data is within [-1,1] and 0 means that we need to normalize
normalized_data = 1;

if flagData == 1 % load household data
    
    load('Individual_Household/x_data');
    %     load('Individual_Household/y_data'); % not normalized in [-1,1]
    load('Individual_Household/y_data_m11'); % normalized in [-1,1]
    n       = size(matX_input, 1); %#ok<*NODEF> % total nr of samples
    d       = size(matX_input, 2); % dimension of the feature vector
    
    n_train = ceil(n * prcntof_data_for_training); % nr of samples for training
    X_train = matX_input(1:n_train, :);
    y_train1= y_sub_metering_1_m11(1:n_train); %y_sub_metering_2; y_sub_metering_3;
    
    if normalized_data == 0
        % since y_train > 1. We modify it
        y_train = y_train1;
        y_train(y_train1<=0) = -1;
        y_train(y_train1>0)  = +1;
    else
        y_train = y_train1;
    end
    
    n_test  = n - n_train;    % nr of test samples
    X_test  = matX_input(n_train+1:end, :);
    y_test1 = y_sub_metering_1_m11(n_train+1:end); %#ok<*COLND> %y_sub_metering_2; y_sub_metering_3;
    
    if normalized_data == 0
        % since y_train > 1. We modify it
        y_test = y_test1;
        y_test(y_test1<=0) = -1;
        y_test(y_test1>0)  = +1;
    else
        y_test = y_test1;
    end
    
    clear matX_input;
    
elseif flagData == 0 % load crimes data
        load('Communities_Crime/x_data');
        load('Communities_Crime/y_data');
        
        n       = size(matX_input, 1); % total nr of samples
        d       = size(matX_input, 2); % dimension of the feature vector
        
        n_train = ceil(n * prcntof_data_for_training); % nr of samples for training
        X_train = matX_input(1:n_train, :);
        y_train = y_data(1:n_train); %y_sub_metering_2; y_sub_metering_3;
        
        n_test  = n - n_train;    % nr of test samples
        X_test  = matX_input(n_train+1:end, :);
        y_test  = y_data(n_train+1:end); %y_sub_metering_2; y_sub_metering_3;
else % generate data
    d       = 10;
    n       = 200;
    data    = logistic_regression_data_generator(n, d);
    X_train = data.x_train.';
    y_train = data.y_train.';
    n_train = numel(y_train);
    
    X_test = data.x_test.';
    y_test = data.y_test.';
    n_test = numel(y_test);
    
end
%% Inputs
% algorithms                = {'GD'; 'SGD'; 'SVRG'};
lambda                    = 0.1; %

nrof_iter                 = 2e3;
nrof_iter_inner_loop      = 20; % SVRG
mini_batch_size           = 10; %round(n*10/100); % for mini-batch SGD
mini_batch_rng_gen        = 1256;

% Create directory to save data
if ~exist('CA3_results', 'dir')
       mkdir('CA3_results')
end
% Create directory to save figures
if ~exist('CA3_figures', 'dir')
       mkdir('CA3_figures')
end
% Open file
fileID = fopen('General_Results.txt','a+');

%% initialize
w3_init     = randn(d,1)-0.5;
W2_init     = randn(d,d)-0.5;
W1_init     = randn(d,d)-0.5;

%% Preliminaries: Cost-function, gradient, and Hessian
% a = @(X, w3, W2, W1) w3*sigmoid(W2*sigmoid(W1*X));
% b = @(X, W2, W1) W2*sigmoid(W1*X);
% c = @(X, W1) W1*X;
% D = @(X, w3, W2, W1) diag((sigmoid(b(X,W2,W1)))^2 .* exp(-b(X, W2, W1)));
% E = @(X, w3, W2, W1) diag((sigmoid(b(X,W2,W1)))^2 .* exp(-b(X, W2, W1)) .* (sigmoid(c(X,W1)))^2 .* exp(-c(X, W1)));
% 
J_cost_L2_logistic_reg                 = @(X, y, N, w3, W2, W1) (1/N)*sum(norm(w3*sigmoid(W2*sigmoid(W1*X)),2)^2);
% grad_J_cost_w3                         = @(X, y, N, w3, W2, W1) 2*(a(X, w3, W2, W1)-y)*sigmoid(b(X, W2, W1));
% grad_J_cost_W2                         = @(X, y, N, w3, W2, W1) 2*(w3*sigmoid(W2*sigmoid(W1*X))-y)*w3'*D(X, w3, W2, W1)*(sigmoid(c(X, W1)))';
% grad_J_cost_W1                         = @(X, y, N, w3, W2, W1) w3*E(X,w3,W2,W1)*X';

%grad_J_cost_L2_logistic_reg_per_sample = @(X, y, N, d, w, lambda) (-X .* repmat((1./(1 + exp(y .* (X*w)))) .* y, [1, d])).' + repmat(lambda*w, [1, N]);
% grad_J_cost_L2_logistic_reg_per_sample = @(X, y, N, w, lambda) bsxfun(@plus, bsxfun(@times, X, -((1./(1 + exp(y .* (X*w))))) .* y).', lambda*w);

% J1                             = J_cost_L2_logistic_reg(X_train, y_train, n_train, w_init, lambda);
%grad_J1                        = grad_J_cost_L2_logistic_reg(X_train, y_train, n_train, w_init, 0.01);
%grad_J1_per_sample             = grad_J_cost_L2_logistic_reg_per_sample(X_train, y_train, n_train, w_init, 0.01);
%J_grad_cost_per_sample         = compute_gradient_per_sample_cost_function(X_train, y_train, n_train, w_init, 0.01);
% [J_hessian_cost, L_ mu]        = compute_hessian_cost_function(X_train, y_train, n_train, w_init, lambda, d);
% 

% z1= @(X, y, N, w3, W2, W1) W1*X;
% a1= @(X, y, N, w3, W2, W1) sigmoid(z1(X, y, N, w3, W2, W1));
% z2= @(X, y, N, w3, W2, W1) W2*a1(X, y, N, w3, W2, W1);
% a2= @(X, y, N, w3, W2, W1) sigmoid(z2(X, y, N, w3, W2, W1));
% z3= @(X, y, N, w3, W2, W1) w3*a2(X, y, N, w3, W2, W1);
% 
% sig3= @(X, y, N, w3, W2, W1) sigmoid(z3(X, y, N, w3, W2, W1))*(1-sigmoid(z3(X, y, N, w3, W2, W1)))*2/N*abs(y-z3(X, y, N, w3, W2, W1));
% sig2= @(X, y, N, w3, W2, W1) sigmoid(z2(X, y, N, w3, W2, W1))*(1-sigmoid(z2(X, y, N, w3, W2, W1)))*w3.'*sig3(X, y, N, w3, W2, W1);
% sig1= @(X, y, N, w3, W2, W1) sigmoid(z1(X, y, N, w3, W2, W1))*(1-sigmoid(z1(X, y, N, w3, W2, W1)))*W2.'*sig2(X, y, N, w3, W2, W1);

% %% Some more inputs for the algorithms
% algo_struct.w3_init                = w3_init;
% algo_struct.W2_init                = W2_init;
% algo_struct.W1_init                = W1_init;
% algo_struct.lambda_reg            = lambda;
% algo_struct.cost_func_handle       = J_cost_L2_logistic_reg;
% % go_struct.grad_handle            = grad_J_cost_L2_logistic_reg;
% % go_struct.grad_per_sample_handle = grad_J_cost_L2_logistic_reg_per_sample;
% algo_struct.nrof_iter              = nrof_iter;
% algo_struct.nrof_iter_inner_loop   = nrof_iter_inner_loop; % valid for SVRG
% algo_struct.step_size              = 1e-5; % fixed value is used if enabled
% algo_struct.step_size_method       = 'adaptive'; %'fixed' 'adaptive' 'adaptive_bb' 'decay'
% algo_struct.mini_batch_size        = mini_batch_size; % mini_batch_size==1 => SGD ... mini_batch_size > 1 => mini_batch SGD
% algo_struct.mini_batch_rng_gen     = mini_batch_rng_gen; % random number


%% Algorithms: core processing

% lambda_vec      = [0 1e-2 1 10 100];%[0, logspace(-4, 2, 3)];

w3_gd = w3_init;
W2_gd = W2_init;
W1_gd = W1_init;

back_gradient_descent(w3_gd, W2_gd, W1_gd, X_train, y_train, 0.5);

end