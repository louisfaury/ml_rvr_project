clc;
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                       %%
%%                      Artificial Dataset 1                             %%
%%                                                                       %%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Constants definition
N               = 100;                %Number of datapoints 
variance        = 0.002;               %Gaussian noise variance
pOutliers       = 0.02;              %Proportion of outliers


%% Generating real function

% X interval
x = -5:0.0001:5;

% Output values
y = sinc(x);

%% Selecting N random datapoints (Uniform)
r = randi([1 length(x)],1,N);           % random points selection
xDataset = x(r)';                       % selected inputs
yDataset = y(r)';                       % selected outputs (noise will be added)
yRefDataset = yDataset;                 % selected outputs reference value


%% Adding noise to outputs

% Gaussian noise
std = sqrt(variance);
yDataset = yDataset + std*randn(N,1);

%% Adding outliers
uOutliersIntensity = (max(y) - min(y))+ 0.2*(max(y) - min(y));
rOutliers    = randi([1 length(xDataset)],round(N*pOutliers),1);
yDataset(rOutliers) = uOutliersIntensity*(rand(round(N*pOutliers),1)-0.5) + median(y);

%% Plotting generated dataset
figure('Name', 'Artificial dataset 1')
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
value1 = '@(x)sinc(x)';

field2 = 'numPoints';
value2 = N;

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

Dataset_1 = struct(field1, value1, field2, value2, field3, value3,...
                    field4, value4, field5, value5, field6, value6,...
                    field7, value7);

save('../datasets/Dataset_1', 'Dataset_1');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                       %%
%%                      Artificial Dataset 2                             %%
%%                                                                       %%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear;

%% Constants definition
N               = 200;                %Number of datapoints 
variance        = 0.005;              %Gaussian noise variance
pOutliers       = 0.01;               %Proportion of outliers


%% Generating real function

% X interval
x = -20:0.01:20;

% Output values
y = exp(-(x-2).^2/4) - exp(-(x+5).^2/10) + exp(-(x+3).^2) + exp(-(x+10).^2/10) - exp(-(x-15).^2/10);

%% Selecting N random datapoints (Uniform)
r = randi([1 length(x)],1,N);           % random points selection
xDataset = x(r)';                       % selected inputs
yDataset = y(r)';                       % selected outputs (noise will be added)
yRefDataset = yDataset;                 % selected outputs reference value


%% Adding noise to outputs

% Gaussian noise
std = sqrt(variance);
yDataset = yDataset + std*randn(N,1);

%% Adding outliers
uOutliersIntensity = (max(y) - min(y))+ 0.2*(max(y) - min(y));
rOutliers    = randi([1 length(xDataset)],N*pOutliers,1);
yDataset(rOutliers) = uOutliersIntensity*(rand(N*pOutliers,1)-0.5) + median(y);

%% Plotting generated dataset
figure('Name', 'Artificial dataset 2')
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
value1 = '@(x)exp(-(x-2).^2/4) - exp(-(x+5).^2/10) + exp(-(x+3).^2) + exp(-(x+10).^2/10) - exp(-(x-15).^2/10)';

field2 = 'numPoints';
value2 = N;

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

Dataset_2 = struct(field1, value1, field2, value2, field3, value3,...
                    field4, value4, field5, value5, field6, value6,...
                    field7, value7);

save('../datasets/Dataset_2', 'Dataset_2');


%% end