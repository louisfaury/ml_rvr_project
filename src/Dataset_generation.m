clc;
clear;
close all;

%% Constants definition
N          = 150;                %Number of datapoints 
var        = 1;                  %Gaussian noise variance
pOutliers  = 0.1;                %Proportion of outliers


%% Generating real function

% X interval
x = -5:0.0001:5;

% Output values
y = 5*sin(x)+(2+sinc(x-1)).^3;

%% Selecting N random datapoints (Uniform)
r = randi([1 length(x)],1,N);
xDataset = x(r)';
yDataset = y(r)';
yRefDataset = yDataset; 

%% Adding noise to outputs

% Gaussian noise
std = sqrt(var);
yDataset = yDataset + std*randn(N,1);

%% Adding outliers
uOutliersIntensity = (max(y) - min(y))+ 0.2*(max(y) - min(y));
rOutliers    = randi([1 length(xDataset)],N*pOutliers,1);
yDataset(rOutliers) = uOutliersIntensity*(rand(N*pOutliers,1)-0.5) + median(y);

%% Plotting generated dataset
figure
hold on;
grid minor;
set(gca, 'FontSize', 14);
plot(x,y, 'r', 'LineWidth', 2);
plot(xDataset, yDataset, 'o', 'MarkerFaceColor', 'b');
xlabel('Input')
ylabel('Output')
legend('Target function', 'Datapoints');

%% Storing in struct and saving
field1 = 'function';
value1 = '5*sin(x)+(2+sinc(x-1)).^3';

field2 = 'numPoints';
value2 = N;

field3 = 'inputs';
value3 = xDataset;

field4 = 'outputs';
value4 = yDataset;

field5 = 'referenceOutputs';
value5 = yRefDataset;

Dataset_1 = struct(field1,value1,field2,value2,field3,value3,...
                    field4,value4,field5, value5);

save('../datasets/Dataset_1', 'Dataset_1');


%% test svr
addpath(genpath('../libsvm'));

% Train model (type svmtrain in command window to see parameters signification)
model = svmtrain(yDataset, xDataset, '-s 4 -t 2 -n 0.5 -g 2');
% Compute prediction
[predicted_label, accuracy, decision_values] = svmpredict(x',x',model);

figure
hold on;
grid minor;
set(gca, 'FontSize', 14);
plot(x,y, 'r', 'LineWidth', 2);
plot(xDataset, yDataset, 'o', 'MarkerFaceColor', 'b');
xlabel('Input')
ylabel('Output')
% Plot prediction
plot(x, predicted_label, 'g', 'LineWidth',2);
%Plot support vectors
plot(xDataset(model.sv_indices), yDataset(model.sv_indices),'o',...
    'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'b');
legend('Target function', 'Datapoints', 'Modeled function', 'Support vectors');


%% test rvr
addpath(genpath('../sparseBayes'));

% Basis functions generation (gaussian kernel)
C = xDataset;
basisWidth = 0.7;
BASIS = exp(-distSquared(xDataset,C)/(basisWidth^2));

% Model train
[PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = SparseBayes('gaussian', BASIS, yDataset);

% Compute infered regression function
w_infer						= zeros(N,1);
w_infer(PARAMETER.Relevant)	= PARAMETER.Value;

% TODO optimize
model = zeros(size(x));
for i=1:size(x,2)
    for j=1:size(PARAMETER.Relevant,1)
        model(i) = model(i) + w_infer(PARAMETER.Relevant(j))*exp(-distSquared(x(i),xDataset(PARAMETER.Relevant(j)))/(basisWidth^2));
    end
end

figure
hold on;
grid minor;
set(gca, 'FontSize', 14);
plot(x,y, 'r', 'LineWidth', 2);
plot(xDataset, yDataset, 'o', 'MarkerFaceColor', 'b');
xlabel('Input')
ylabel('Output')
plot(x, model, 'g', 'LineWidth',2);
plot(xDataset(PARAMETER.Relevant), yDataset(PARAMETER.Relevant),'o',...
    'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'b');
legend('Target function', 'Datapoints', 'Modeled function', 'Relevant vectors');


%% end