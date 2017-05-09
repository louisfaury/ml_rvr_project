function model = rvr(ds,k,params,f);
% ============= HEADER ============= %
% \brief   ? Calls sparseBayes Relevance Machine Regression methods
% \param   ? ds <- dataset 
%          ? k  <- kernel { 
%                           'name'
%                           'params'
%                       }
%          ? params  <- parameters {
%                           'alpha'
%                           'beta'
%                       }
%          ? f  <- plot option
% \returns ? Object with sparseBayes RVM output and 'predict' function
% ============= HEADER ============= %

% variable definitions 
inputs  = ds.inputs;
targets = ds.outputs;
true_f  = str2func(ds.function);
xmin    = ds.minInput;
xmax    = ds.maxInput;
n       = ds.numPoints;

%Compute basis (design) matrix
switch k.name 
    case 'linear'
       BASIS = inputs * inputs';
    case 'polynomial'
       BASIS = (inputs * inputs').^k.params.degree;
    case 'rbf'
       C = inputs; 
       BASIS = exp(-distSquared(inputs,C)/(k.params.width^2));
    otherwise 
        error('Unknown kernel');
end

OPTIONS     = SB2_UserOptions();
SETTINGS    = SB2_ParameterSettings('Beta',params.beta,'Alpha',params.alpha * ones(ds.numPoints,1),'Relevant', (1:n)');

% run
[Parameter, Hyperparameter, Diagnostic] = ...
    SparseBayes('Gaussian', BASIS, targets, OPTIONS, SETTINGS);

w_infer						= zeros(n,1);
w_infer(Parameter.Relevant)	= Parameter.Value;

switch k.name 
    case 'linear'
       predict = @(x) w_infer(Parameter.Relevant)' * inputs(Parameter.Relevant,:) * x;
    case 'polynomial'
       predict = @(x) w_infer(Parameter.Relevant)' * ((inputs(Parameter.Relevant,:) * x).^k.params.degree);
    case 'rbf'
       predict = @(x) w_infer(Parameter.Relevant)' * exp(-distSquared(inputs(Parameter.Relevant,:),x)/(k.params.width^2));
    otherwise 
       error('Unknown kernel');
end

model = struct('Parameter', Parameter, 'Hyperparameter', Hyperparameter, 'Diagnostic', Diagnostic, 'predict', predict);

% plot
if (f)
   % x = (xmin:0.1:xmax)';
     x = (-10:0.1:10)';
   y = true_f(x);
    
    label = predict(x);

    figure
    hold on;
    grid minor;
    set(gca, 'FontSize', 14);
    
    % builds the predictive distribution 
    max = intmax;
    a = double(max)*ones(n+1,1);
    a(Parameter.Relevant) = Hyperparameter.Alpha;
    A = diag(a);
    K = compute_gram_matrix(k,inputs,inputs);
    Sigma = inv(A + Hyperparameter.beta*(K'*K));
    % m = Hyperparameter.beta * Sigma *( K' * targets); % can be used for prediction
    Kx = compute_gram_matrix(k,x,inputs);
    proj_std = sqrt(1/Hyperparameter.beta*ones(size(x)) +  diag(Kx * Sigma * Kx'));
    % plot support vectors
    p1 = scatter(inputs(Parameter.Relevant), targets(Parameter.Relevant),80*ones(size(inputs((Parameter.Relevant)))),'o','MarkerEdgeColor', 'g', 'MarkerFaceColor', 'w','LineWidth',1.5);
    scatter(inputs(Parameter.Relevant), targets(Parameter.Relevant),30*ones(size(inputs((Parameter.Relevant)))),'MarkerFaceColor','r','MarkerFaceAlpha',0.8,'MarkerEdgeColor','r','MarkerEdgeAlpha',0.8);
    % plot dataset
    p2 = scatter(inputs, targets,30*ones(size(inputs)),'MarkerFaceColor','r','MarkerFaceAlpha',0.4,'MarkerEdgeColor','r','MarkerEdgeAlpha',0.4);
    % plot reference function
    p3 = plot(x,y, 'k', 'LineWidth', 2);
    % plot prediction
    p4 = plot(x, label, '-','Color',[0.3 0.6 1], 'LineWidth',2.2);  
    % plot covariance
    %p5 = plot(x,label+proj_std', '-.','Color',[0.3 0.6 1], 'LineWidth',2.2);
    %area(x,label+proj_std',);
    %plot(x,label-proj_std', '-.','Color',[0.3 0.6 1], 'LineWidth',2.2);
    p5 = jbfill(x',label+proj_std',label-proj_std','b',[1 1 1],0,0.1);

    xlabel('Input')
    ylabel('Output')
    legend([p1,p2,p3,p4,p5],{'Support vectors', 'Datapoints','Target function', 'Modeled function','Predictive distribution (std)'});
    title(strcat('Result of RVR for dataset',{' '}, ds.name))
end

end