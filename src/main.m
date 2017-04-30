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

%% Support Vector Regression 
% Define kernel, hp
kernelstr = 'rbf'; % 'gaussian', 'polynomial', 'linear'
switch kernelstr
    case 'rbf'
        width = 2;
        params = struct('width',width);
    case 'polynomial'
        degree = 2;
        params = struct('degree',degree);
    otherwise
        params = [];
end
kernel = struct('name',kernelstr,'params',params);
% defines svr type
type = 'C'; % 'C', 'nu'
switch type
    case 'C'
        C = 100;
        eps = 0.3;
        params = struct('C',C,'eps',eps);
    case 'nu'
        C = 1;
        nu = 0.5;
        params = struct('C',C,'nu',nu);
    otherwise
        error('Unknown SVR type');
end
type = struct('type',type,'params',params);
% call SVR 
plot_flag = 1;
model = svr(Dataset,kernel,type,plot_flag);



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
    

    
    
    
%% IDEAS : 
% -SVR : try to 'impose' through nu-svr the same number of RV as in RVM to compare performance at equel level of sparsity  
% - General : before doing cross-validation, compare accuracy on full
% training sets ? 