clc;
clear; 
close all;

%% adding library to path
addpath(genpath('../lib/sparse_bayes'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                       %%
%%                      Artificial Dataset 1                             %%
%%                                                                       %%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;

%% loading dataset
load('../datasets/dataset_sinc');

xDataset    = Dataset_1.inputs;
yDataset    = Dataset_1.outputs;
yRefDataset = Dataset_1.referenceOutputs;
func        = str2func(Dataset_1.function);
xMin        = Dataset_1.minInput;
xMax        = Dataset_1.maxInput;
N           = Dataset_1.numPoints;

%% retrieving reference function
x = xMin:0.01:xMax;
y = func(x);

%% basis functions generation (gaussian kernel)
c = xDataset;
basisWidth = 0.5;
basis = exp(-distSquared(xDataset,c)/(basisWidth^2));

%% train model
[Parameter, HyperParameter, Diagnostic] = SparseBayes('gaussian', basis, yDataset);

%% compute infered regression function
w_infer						= zeros(N,1);
w_infer(Parameter.Relevant)	= Parameter.Value;

% TODO optimize
model = zeros(size(x));
for i=1:size(x,2)
    for j=1:size(Parameter.Relevant,1)
        model(i) = model(i) + w_infer(Parameter.Relevant(j))*exp(-distSquared(x(i),xDataset(Parameter.Relevant(j)))/(basisWidth^2));
    end
end


%% plot results
figure
hold on;
grid minor;
set(gca, 'FontSize', 14);
% plot reference function
plot(x,y, 'r', 'LineWidth', 2);
% plot dataset
plot(xDataset, yDataset, 'o', 'MarkerFaceColor', 'b');
% plot retrieved model
plot(x, model, 'g', 'LineWidth',2);
% plot relevant vectors
plot(xDataset(Parameter.Relevant), yDataset(Parameter.Relevant),'o',...
    'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'b');
legend('Target function', 'Datapoints', 'Modeled function', 'Relevant vectors');
xlabel('Input')
ylabel('Output')