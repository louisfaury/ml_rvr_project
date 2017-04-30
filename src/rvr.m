function model = rvr(ds,k,params,f);
% ============= HEADER ============= %
% \brief   ? Calls libsvm Support Vector Regression methods 
% \param   ? ds <- dataset 
%          ? k  <- kernel { 
%                           'name'
%                           'params'
%                       }
%          ? m  <- method : 'C' for C-svr, 'nu' for nu-SVR with hp 
%          ? f  <- plot option
% \returns ? index of relevance vectors as well as respective coeffs
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
SETTINGS    = SB2_ParameterSettings('Beta',params.beta,'Alpha',params.alpha,'Relevant', (1:n)');

% run
[Parameter, Hyperparameter, Diagnostic] = ...
    SparseBayes('Gaussian', BASIS, targets, OPTIONS, SETTINGS);

model = struct('Parameter', Parameter, 'Hyperparameter', Hyperparameter, 'Diagnostic', Diagnostic);
% plot
if (f)
    x = (xmin:0.1:xmax)';
    y = true_f(x);
    
    w_infer						= zeros(n,1);
    w_infer(Parameter.Relevant)	= Parameter.Value;

    switch k.name 
    case 'linear'
       k_mat = inputs(Parameter.Relevant,:) * x;
    case 'polynomial'
       k_mat = (inputs(Parameter.Relevant,:) * x).^k.params.degree;
    case 'rbf'
       k_mat = exp(-distSquared(inputs(Parameter.Relevant,:),x)/(k.params.width^2));
    otherwise 
        error('Unknown kernel');
    end
    
    label = w_infer(Parameter.Relevant)' * k_mat;
    
    figure
    hold on;
    grid minor;
    set(gca, 'FontSize', 14);
    
    % plot support vectors
    p1 = scatter(inputs(Parameter.Relevant), targets(Parameter.Relevant),80*ones(size(inputs((Parameter.Relevant)))),'o','MarkerEdgeColor', 'g', 'MarkerFaceColor', 'w','LineWidth',1.5);
    scatter(inputs(Parameter.Relevant), targets(Parameter.Relevant),30*ones(size(inputs((Parameter.Relevant)))),'MarkerFaceColor','r','MarkerFaceAlpha',0.8,'MarkerEdgeColor','r','MarkerEdgeAlpha',0.8);
    % plot dataset
    p2 = scatter(inputs, targets,30*ones(size(inputs)),'MarkerFaceColor','r','MarkerFaceAlpha',0.4,'MarkerEdgeColor','r','MarkerEdgeAlpha',0.4);
    % plot reference function
    p3 = plot(x,y, 'k', 'LineWidth', 2);
    % plot prediction
    p4 = plot(x, label, '-.','Color',[0.3 0.6 1], 'LineWidth',2.2);  
    
    xlabel('Input')
    ylabel('Output')
    legend([p1,p2,p3,p4],{'Support vectors', 'Datapoints','Target function', 'Modeled function'});
    title(strcat('Result of RVR for dataset',{' '}, ds.name))
end

end