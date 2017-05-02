%% Advanced Machine Learning Coding Project
% \brief  : Main script for the Relevance Vector Machine Project
% \author : Gregoire Gallois-Montbrun
%           Hadrien Hendrix
%           Louis Faury
% \date : 15/04/2017

clear all;
close all;
clc; 

%% add paths 
addpath(genpath('../lib/libsvm'));
addpath(genpath('../lib/sparse_bayes'));
addpath(genpath('../datasets'));

%% load 
name = 'sinc'; % 'sinc', 'online_views'
load(strcat('dataset_',name,'.mat')); 

%% Prepare cross validation
models = [];

%% Support Vector Regression 
% Define kernel, hp
kernelstr = 'rbf'; % 'gaussian', 'polynomial', 'linear'
kernel = generate_kernel(kernelstr, 2);

% defines svr type
type = 'C'; % 'C', 'nu'
C = 100;
param = 0.1; %represents either epsilon or nu depending on the type

% call SVR 
plot_flag = 1;
model = generate_SVR(type, kernel, C, param, 'SVR');
model = train_model(Dataset,model,plot_flag);

%% Relevance Vector Regression 
% Define kernel, hp
kernelstr = 'rbf'; % 'gaussian', 'polynomial', 'linear'
width = 2;
kernel = generate_kernel(kernelstr, width);

% define alpha and beta
% Assume alpha is a numerical value and is the same for each point
alpha = 0.1;
beta = 0.1;

model = generate_RVR(kernel,alpha,beta,'RVR');
% call RVR 
plot_flag = 1;
model = train_model(Dataset,model,plot_flag);

% First round, vizu 
    % 2. call baseline functions for svr and rvr -> deux fonctions 
    %                                               params : dataset, options (hp) 
    %                                               output : plot handle
    % 3. post process (plots, hp), compute generalization 
    
    % [index_of_relevance, coeff_of_relevance, iterations_number,
    % learning_curve(?)] = my_ml_function(dataset,options)
    % 
    
    % 4. plot !
 
%% Cross-validation, evaluative plots 
% Cross-validation 
    % 4. Call cv function 
    %           params : f-fold and training test ratio 
    %           output : metric statisitics 
    gk2 = generate_kernel('rbf',2);
    
    models = [];
    models = [models generate_SVR('C',gk2,100,0.3, 'C SVR 100')];
    models = [models generate_SVR('C',gk2,1,0.3, 'C SVR 1')];
    models = [models generate_RVR(gk2, 0.1, 0.1, 'RVR alpha 0.1')];
    models = [models generate_RVR(gk2, 0.1, 10, 'RVR alpha 10')];
    
    cross_validate(Dataset, models, 5, 0.8);

    
    
    
%% IDEAS : 
% -SVR : try to 'impose' through nu-svr the same number of RV as in RVM to compare performance at equel level of sparsity  
% - General : before doing cross-validation, compare accuracy on full
% training sets ? 