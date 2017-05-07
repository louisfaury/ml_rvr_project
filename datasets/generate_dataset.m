clc;
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                       %%
%%                      Artificial Dataset 1                             %%
%%                                                                       %%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Constants definition
n               = 100;       % Number of datapoints 
variance        = 0.01;    % Gaussian noise variance
pOutliers       = 0;     % Proportion of outliers

%% Generating real function
% X interval
x = -5:0.0001:5;
% Output values
y = sinc(x);

%% Selecting N random datapoints (Uniform)
r           = randi([1 length(x)],1,n);           % random points selection
xDataset    = x(r)';                       % selected inputs
yDataset    = y(r)';                       % selected outputs (noise will be added)
yRefDataset = yDataset;                 % selected outputs reference value


%% Adding noise to outputs
% Gaussian noise
std      = sqrt(variance);
yDataset = yDataset + std*randn(n,1);

%% Adding outliers
uOutliersIntensity  = (max(y) - min(y))+ 0.2*(max(y) - min(y));
rOutliers           = randi([1 length(xDataset)],round(n*pOutliers),1);
yDataset(rOutliers) = uOutliersIntensity*(rand(round(n*pOutliers),1)-0.5) + median(y);

%% Plotting generated dataset
figure('Name', 'Artificial dataset 1')
hold on;
grid minor;
set(gca,'FontSize', 14);
plot(x,y, 'r', 'LineWidth', 2);
plot(xDataset, yDataset, 'o', 'MarkerFaceColor', 'b');
xlabel('Input')
ylabel('Output')
legend('Target function', 'Datapoints');

%% Storing in struct and saving
field1 = 'function';
value1 = '@(x)sinc(x)';

field2 = 'numPoints';
value2 = n;

field3 = 'inputs';
value3 = xDataset;

field4 = 'outputs';
value4 = yDataset;

field5 = 'referenceOutputs';
value5 = yRefDataset;

field6 = 'minInput';
value6 = min(x);

field7 = 'maxInput';
value7 = max(x);

field8 = 'name';
value8 = 'sinc';

Dataset = struct(field1, value1, field2, value2, field3, value3,...
                    field4, value4, field5, value5, field6, value6,...
                    field7, value7, field8, value8);

save('../datasets/dataset_sinc', 'Dataset');