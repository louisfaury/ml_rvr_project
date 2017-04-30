%% Advanced Machine Learning Coding Project
% \brief : Main script for the Relevance Vector Machine Project
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

%% call ml functions
% Define kernel, hp
kernelstr = 'gaussian'; % 'gaussian', 'poly2' 
switch kernelstr
    case 'gaussian'
        width = 0.3;
        params = struct('width',width);
    case 'poly2'
        params = [];
    otherwise
        error('Unknown kernel string form');
end
kernel = struct('name',kernelstr,'params',params);
% Call SVR 


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
    
