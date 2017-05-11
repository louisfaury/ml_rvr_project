%% Advanced Machine Learning Coding Project
% \brief  : Main script for the Relevance Vector Machine Project
% \author : Gregoire Gallois-Montbrun
%           Hadrien Hendrix
%           Louis Faury
% \date : 15/04/2017

clear;
close all;
clc;

%% latex interpreter for figures
set(0, 'defaulttextInterpreter', 'latex');

%% add paths
addpath(genpath('../lib/libsvm'));
addpath(genpath('../lib/sparse_bayes'));
addpath(genpath('../datasets'));
addpath(genpath('../lib'))

%% load
name = 'airfoils'; % 'sinc', 'online_views', 'airfoils'
load(strcat('dataset_',name,'.mat'));

%% Support Vector Regression 
% Define kernel, hp
kernelstr = 'rbf'; % 'rbf', 'polynomial', 'linear'
kernel = generate_kernel(kernelstr, 0.25);

% defines svr type
type = 'nu'; % 'C', 'nu'
C = 200;
param = 0.15; %represents either epsilon or nu depending on the type

% call SVR 
plot_flag = 1;
model = generate_SVR(type, kernel, C, param, 'SVR');
model = train_model(Dataset,model,plot_flag);

%% Relevance Vector Regression
%Define kernel, hp
% kernelstr = 'rbf'; % 'gaussian', 'polynomial', 'linear'
% width = 1.35;
% kernel = generate_kernel(kernelstr, width);
% 
% % define alpha and beta
% % Assume alpha is a numerical value and is the same for each point
% alpha = 0.1;
% beta = 0.1;
% 
% model = generate_RVR(kernel,alpha,beta,'RVR');
% % call RVR
% plot_flag = 0;
% model = train_model(Dataset,model,plot_flag);

%% Cross-validation, evaluative plots
% Cross-validation
% gk2 = generate_kernel('rbf',2);
%
% models = [];
% models = [models generate_SVR('C',gk2,100,0.3, 'C SVR 100')];
% models = [models generate_SVR('C',gk2,1,0.3, 'C SVR 1')];
% models = [models generate_RVR(gk2, 0.1, 0.1, 'RVR alpha 0.1')];
% models = [models generate_RVR(gk2, 0.1, 10, 'RVR alpha 10')];
%
% cross_validate(Dataset, models, 5, 0.8, 1); % prettier, box plot

%% Grid search for nu-SVR
nfold = 10;
ttratio = 0.5;

type = 'nu';
% Define kernel, hp
kernelstr = 'rbf'; % 'gaussian', 'polynomial', 'linear'
% Define range of hyperparameters

nu      = linspace(0.001, 0.1, 20);
C       = logspace(-1, 1.5, 20);
sigma   = logspace(-2, 1, 20);
% Perform gridsearch
grid_search_cv(Dataset, 'SVR', ttratio, nfold, kernelstr, type, nu, C, sigma);

%Optimal params for:
%MSE: sigma = 1.1288, nu = 0.19737, C = 5.1348
%BIC: sigma = 1.1288, nu = 0.032263, C = 3.7927
%% Grid search for C-SVR
<<<<<<< HEAD
nfold = 50;
type = 'C';
ttratio = 0.75;
% Define kernel
kernelstr = 'rbf'; % 'gaussian', 'polynomial', 'linear'
% Define range of hyperparameters
eps     = logspace(-2, 1, 30);
C       = logspace(-1, 1.5, 30);
sigma   = logspace(-2, 1, 30);
% Perform gridsearch
grid_search_cv(Dataset, 'SVR', ttratio, nfold, kernelstr, type, eps, C, sigma);

%Optimal params for:
%MSE: sigma = 0.92367, eps = 0.085317, C = 0.88772
%BIC: sigma = 1.1721, eps = 0.17433, C = 2.9209
%% Grid search for RVR
% ttratio = 0.5;
% nfold   = 10;
% % Define kernel, hp
% kernelstr = 'rbf'; % 'gaussian', 'polynomial', 'linear'
% % Define range of hyperparameters
% sigma   = logspace(-0.5, 1, 20);
% % Perform gridsearch
% grid_search_cv(Dataset, 'RVR', ttratio, nfold, kernelstr, sigma);

%Optimal params for:
%MSE: sigma = 0.56
%BIC: sigma = 1.28

% define kernel
% grid_search_cv() -> C-SVR  : eps, C, sigma -> colormap two versus one opt
%                  -> nu-SVR : nu, C, sigma -> colormap two versus one opt
%                  -> RVR    : sigma -> 1d plot
% with MSE, AIC & BIC

% kernel engineer
% design polynomial (sum, ..) kernel

%% Sparsity vs MSE curves
k_rvr_BIC = generate_kernel('rbf',1.28);
k_rvr_MSE = generate_kernel('rbf',0.56);
k_csvr_BIC = generate_kernel('rbf',1.1721);
k_csvr_MSE = generate_kernel('rbf',0.92367);
k_nusvr_BIC = generate_kernel('rbf',1.1288);
k_nusvr_MSE = generate_kernel('rbf',1.1288);

%MSE: sigma = 1.1288, nu = 0.19737, C = 5.1348
%BIC: sigma = 1.1288, nu = 0.032263, C = 3.7927

models = [];
%Optimal RVR model for BIC
models = [models generate_RVR(k_rvr_BIC,100,0.3, 'RVR MSE/BIC')];
%Optimal RVR model for MSE
%models = [models generate_RVR(k_rvr_MSE,1,0.3, 'RVR BIC')];
%Optimal nu SVR model for BIC
models = [models generate_SVR('nu',k_nusvr_BIC, 3.7927, 0.032263, 'NU SVR BIC')];
%Optimal nu SVR model for MSE
models = [models generate_SVR('nu',k_nusvr_MSE, 5.1348, 0.19737, 'NU SVR MSE')];
%Optimal C SVR for BIC
models = [models generate_SVR('C',k_csvr_BIC, 2.9209, 0.17433, 'C SVR BIC')];
%Optimal C SVR for MSE
models = [models generate_SVR('C',k_csvr_MSE, 0.88772, 0.085317, 'C SVR MSE')];
 
ttratio = 0.75;
nfold   = 50;

sparsity_vs_mse(Dataset, models, nfold, ttratio)

% computation time, memory (see doc, tic toc)
%% IDEAS :
% -SVR : try to 'impose' through nu-svr the same number of RV as in RVM to compare performance at equel level of sparsity
