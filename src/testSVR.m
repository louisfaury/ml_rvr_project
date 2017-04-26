clc;
clear; 
close all;

%% adding library to path
addpath(genpath('../lib/libsvm'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                       %%
%%                      Artificial Dataset 1                             %%
%%                                                                       %%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;

%% loading dataset
load('../datasets/dataset_sinc');

xDataset    = Dataset.inputs;
yDataset    = Dataset.outputs;
yRefDataset = Dataset.referenceOutputs;
func        = str2func(Dataset.function);
xMin        = Dataset.minInput;
xMax        = Dataset.maxInput;
N           = Dataset.numPoints;

%% retrieving reference function
x = xMin:0.01:xMax;
y = func(x);


%% train model (type svmtrain in command window to see parameters signification)
model = svmtrain(yDataset, xDataset, '-s 4 -t 2 -n 0.5 -g 1');

%% compute prediction
[predictedOutput, accuracy, decisionValues] = svmpredict(x',x',model);

%% plot results
figure
hold on;
grid minor;
set(gca, 'FontSize', 14);

% plot reference function
plot(x,y, 'r', 'LineWidth', 2);
% plot dataset
plot(xDataset, yDataset, 'o', 'MarkerFaceColor', 'b');
% plot prediction
plot(x, predictedOutput, 'g', 'LineWidth',2);
% plot support vectors
plot(xDataset(model.sv_indices), yDataset(model.sv_indices),'o',...
    'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'b');

xlabel('Input')
ylabel('Output')
legend('Target function', 'Datapoints', 'Modeled function', 'Support vectors');
title('Result of SVR for dataset 1')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                       %%
%%                      Artificial Dataset 2                             %%
%%                                                                       %%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;

%% loading dataset
load('../datasets/Dataset_2');

xDataset    = Dataset_2.inputs;
yDataset    = Dataset_2.outputs;
yRefDataset = Dataset_2.referenceOutputs;
func        = str2func(Dataset_2.function);
xMin        = Dataset_2.minInput;
xMax        = Dataset_2.maxInput;
N           = Dataset_2.numPoints;

%% retrieving reference function
x = xMin:0.01:xMax;
y = func(x);


%% train model (type svmtrain in command window to see parameters signification)
model = svmtrain(yDataset, xDataset, '-s 4 -t 2 -n 0.1 -g 1');

%% compute prediction
[predictedOutput, accuracy, decisionValues] = svmpredict(x',x',model);

%% plot results
figure
hold on;
grid minor;
set(gca, 'FontSize', 14);

% plot reference function
plot(x,y, 'r', 'LineWidth', 2);
% plot dataset
plot(xDataset, yDataset, 'o', 'MarkerFaceColor', 'b');
% plot prediction
plot(x, predictedOutput, 'g', 'LineWidth',2);
% plot support vectors
plot(xDataset(model.sv_indices), yDataset(model.sv_indices),'o',...
    'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'b');

xlabel('Input')
ylabel('Output')
legend('Target function', 'Datapoints', 'Modeled function', 'Support vectors');
title('Result of SVR for dataset 2')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                       %%
%%                      Online Popularity Dataset                        %%
%%                                                                       %%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;

%% loading dataset
load('../datasets/Dataset_onlineNewsPop');

xDataset    = Dataset.inputs;
yDataset    = Dataset.outputs;
N           = Dataset.numPoints;

%% training ratio
trainingRatio = 0.4;

%% standardization
sxDataset = (xDataset-mean(xDataset))./std(xDataset);
syDataset = (yDataset-mean(yDataset))./std(yDataset);

%% select sub-dataset for training tractability
subsample = randi([1 N],1,round(N*trainingRatio));
sxTrain   = sxDataset(subsample,:);
syTrain   = syDataset(subsample,:);
    
%% train model (type svmtrain in command window to see parameters signification)
tic
model = svmtrain(syTrain, sxTrain, '-s 4 -t 2 -n 0.05 -g 0.05');
toc

%% compute prediction
[predictedOutput, accuracy, decisionValues] = svmpredict(syDataset, sxDataset, model);
predictedViews = round(predictedOutput.*std(yDataset)+ mean(yDataset));
%% end