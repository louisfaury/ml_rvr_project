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
name = 'sinc'; % 'sinc', 'online_views'
load(strcat('dataset_',name,'.mat')); 

%% Prepare cross validation
models = [];

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
% Define kernel, hp
kernelstr = 'rbf'; % 'gaussian', 'polynomial', 'linear'
width = 0.73;
kernel = generate_kernel(kernelstr, width);

% define alpha and beta
% Assume alpha is a numerical value and is the same for each point
alpha = 0.1;
beta = 0.1;

model = generate_RVR(kernel,alpha,beta,'RVR');
% call RVR 
plot_flag = 1;
model = train_model(Dataset,model,plot_flag);
 
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


% %% Grid search for nu-SVR
% nfold = 50;
% ttratio = 0.75;
% type = 'nu';
% % Define kernel, hp
% kernelstr = 'rbf'; % 'gaussian', 'polynomial', 'linear'
% % Define range of hyperparameters
% nu      = linspace(0.01, 0.9, 20);
% C       = logspace(-1, 1.5, 20);
% sigma   = logspace(-2, 1, 20);
% % Perform gridsearch
% grid_search_cv(Dataset, 'SVR', ttratio, nfold, kernelstr, type, nu, C, sigma);


%% Grid search for C-SVR
% nfold = 50;
% type = 'C';
% ttratio = 0.75;
% % Define kernel
% kernelstr = 'rbf'; % 'gaussian', 'polynomial', 'linear'
% % Define range of hyperparameters
% eps     = logspace(-2, 1, 30);
% C       = logspace(-1, 1.5, 30);
% sigma   = logspace(-2, 1, 30);
% % Perform gridsearch
% grid_search_cv(Dataset, 'SVR', ttratio, nfold, kernelstr, type, eps, C, sigma);

%% Grid search for RVR
ttratio = 0.75;
nfold   = 50;
% Define kernel, hp
kernelstr = 'rbf'; % 'gaussian', 'polynomial', 'linear'
% Define range of hyperparameters
sigma   = logspace(-1.5, 1, 15);
% Perform gridsearch
grid_search_cv(Dataset, 'RVR', ttratio, nfold, kernelstr, sigma);

% define kernel
% grid_search_cv() -> C-SVR  : eps, C, sigma -> colormap two versus one opt
%                  -> nu-SVR : nu, C, sigma -> colormap two versus one opt 
%                  -> RVR    : sigma -> 1d plot 
% with MSE, AIC & BIC 

% kernel engineer 
% design polynomial (sum, ..) kernel 

% computation time, memory (see doc, tic toc)
%% IDEAS : 
% -SVR : try to 'impose' through nu-svr the same number of RV as in RVM to compare performance at equel level of sparsity  
