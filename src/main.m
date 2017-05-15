%% ADVANCED MACHINE LEARNING CODING PROJECT
% \brief  : Main script for the Relevance Vector Machine Project
% \author : Gregoire Gallois-Montbrun
%           Hadrien Hendrix
%           Louis Faury
% \date : 15/04/2017

clear;
close all;
clc;
set(0, 'defaulttextInterpreter', 'latex');
addpath(genpath('../lib/libsvm'));
addpath(genpath('../lib/sparse_bayes'));
addpath(genpath('../datasets'));
addpath(genpath('../lib'))


%% s1.LOAD DATASET
% Loads 'sinc' dataset
 name = 'sinc';
 plot_flag = 1;
% Loads 'airfoils' dataset
%name = 'airfoils';
%plot_flag = 0;

load(strcat('dataset_',name,'.mat'));

%% s2.SUPPORT VECTOR REGRESSION -- uncomment for a test run (train+predict) 
% % Defines kernel and its hyperparameters
% kernelstr = 'rbf';                              % 'rbf', 'polynomial', 'linear'
% kernel = generate_kernel(kernelstr, 1.1288);
% % Defines svr type
% type = 'nu';                                    % 'C', 'nu'
% C = 5;
% param = 0.19;                                   % param = epsilon or nu depending on the type
% % Calls SVR
% 
% model = generate_SVR(type, kernel, C, param, 'SVR');
% model = train_model(Dataset,model,plot_flag);


%% s3.RELEVANCE VECTOR REGRESSION -- uncomment for a test run (train+predict)
% Defines base functions and its hyperparameters
% kernelstr = 'rbf';                             % 'rbf', 'polynomial', 'linear'
% width = 1.35;
% kernel = generate_kernel(kernelstr, width);
% alpha = 0.1;                                    % initial alpha
% beta = 0.1;                                     % initial beta 
% model = generate_RVR(kernel,alpha,beta,'RVR');
% % call RVR
% model = train_model(Dataset,model,plot_flag);


%% s4.BIC METRIC TESTING 
% % Shows the tradeoff found by our BIC measure on the artificial dataset,
% % on a nu-SVR example 
% models = [];
% k_nusvr_BIC = generate_kernel('rbf',1.1288);
% k_nusvr_MSE = generate_kernel('rbf',1.1288);
% 
% if (strcmp('sinc',name))
%     % Optimal nu-SVR model for BIC
%     models = [models generate_SVR('nu',k_nusvr_BIC, 3.7927, 0.032263, 'BIC $\star$')];
%     % Optimal nu-SVR model for MSE
%     models = [models generate_SVR('nu',k_nusvr_MSE, 5.1348, 0.19737, 'MSE $\star$')];
%     % Optimal nu-SVR model for BIC k^2 ln N
%     models = [models generate_SVR('nu',k_nusvr_MSE, 1.5283, 0.037474, 'BIC squared')];
%     % Optimal nu-SVR model for BIC sqrt(k) ln N
%     models = [models generate_SVR('nu',k_nusvr_MSE, 17.2521, 0.084368, 'BIC sqrt')];
%     % Optimal nu-SVR model for BIC k ln k
%     models = [models generate_SVR('nu',k_nusvr_MSE, 2.8014, 0.068737, 'BIC k lnk')];
% 
%     % Some arbitrary nu-SVR models 
%     models = [models,generate_inrange_model('nu',0.5,1.5,1,10,0.02,0.15)];
%     
%     sparsity_vs_mse(Dataset, models,10,0.75,[1,2,3,4,5]);
% end

%% s5.GRID-SEARCH CROSS-VALIDATION FOR NU-SVR (rbf kernel)
n_fold = 10;        % folds
tt_ratio = 0.5;     % training-testing ratio
% Defines grid search range : 
nu      = linspace(0.001, 1, 5);
C       = logspace(-1, 3, 5);
sigma   = logspace(-2, 1, 5);
grid_search_cv(Dataset, 'SVR', tt_ratio, n_fold, 'rbf', 'nu', nu, C, sigma); % performs grid search
% Optimal params are :
%       MSE: sigma = 1.1288, nu = 0.19737, C = 5.1348
%       BIC: sigma = 1.1288, nu = 0.032263, C = 3.7927
%       bic k squared: sigma = 1.1288, nu = 0.037474 C = 1.5283
%       bic sqrt k                  0.084368, 17.2521
%       bic k ln k              0.068737, 2.8014

%% s6.GRID-SEARCH CROSS-VALIDATION FOR eps-SVR (rbf kernel)
n_fold = 50;        % folds
tt_ratio = 0.5;     % training-testing ratio 
% Defines grid search range :
% eps     = logspace(-2, 1, 30);
% C       = logspace(-1, 1.5, 30);
% sigma   = logspace(-2, 1, 30);
% grid_search_cv(Dataset, 'SVR', tt_ratio, n_fold, 'rbf', 'C', eps, C, sigma); % performs grid search
% % Optimal params for (artificial dataset):
% %       MSE: sigma = 0.92367, eps = 0.085317, C = 0.88772
% %       BIC: sigma = 1.1721, eps = 0.17433, C = 2.9209


%% s7.GRID-SEARCH CROSS-VALIDATION FOR RVR (rbf kernel)
% tt_ratio = 0.5;  % trainig_testing ratio 
% n_fold   = 10;   % fold 
% % Defines range of hyperparameters
% sigma  = logspace(-0.5, 1, 20);
% grid_search_cv(Dataset, 'RVR', tt_ratio, n_fold, 'rbf', sigma);
% % Optimal params for:
% %       MSE: sigma = 0.56
% %       BIC: sigma = 1.28


%% s8.PLOT 'ROC' CRUVES : SPARSITY VS MSE
% k_rvr_BIC   = generate_kernel('rbf',1.28);
% k_rvr_MSE   = generate_kernel('rbf',0.56);
% k_csvr_BIC  = generate_kernel('rbf',1.1721);
% k_csvr_MSE  = generate_kernel('rbf',0.92367);
% k_nusvr_BIC = generate_kernel('rbf',1.1288);
% k_nusvr_MSE = generate_kernel('rbf',1.1288);
% models = [];
% % Optimal RVR model for BIC
% models = [models generate_RVR(k_rvr_BIC,100,0.3, 'RVR MSE/BIC')];
% % Optimal nu SVR model for BIC
% models = [models generate_SVR('nu',k_nusvr_BIC, 3.7927, 0.032263, '$\nu$-SVR BIC')];
% %Optimal nu SVR model for MSE
% models = [models generate_SVR('nu',k_nusvr_MSE, 5.1348, 0.19737, '$\nu$-SVR MSE')];
% %Optimal C SVR for BIC
% models = [models generate_SVR('C',k_csvr_BIC, 2.9209, 0.17433, '$\varepsilon$-SVR BIC')];
% %Optimal C SVR for MSE
% models = [models generate_SVR('C',k_csvr_MSE, 0.88772, 0.085317, '$\varepsilon$-SVR MSE')];
% % Generates plots : 
% tt_ratio = 0.75;
% n_fold   = 50;
% sparsity_vs_mse(Dataset, models,50,0.75); % 50-fold with 0.75 training test ratio