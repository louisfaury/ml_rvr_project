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
% First round, vizu 
    % 2. call baseline functions for svr and rvr -> deux fonctions 
    %                                               params : dataset, options (hp) 
    %                                               output : plot handle
    % 3. post process (plots, hp), compute generalization 
    
 
%% Cross-validation, evaluative plots 
% Cross-validation 
    % 4. Call cv function 
    %           params : f-fold and training test ratio 
    %           output : metric statisitics 
    
