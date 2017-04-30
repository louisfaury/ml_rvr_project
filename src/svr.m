function model = svr(ds,k,m,f);
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

% param settings 
ctrl_str = '';
switch m.type
    case 'C'
        m_str = strcat('-s 3 -c',{' '},num2str(m.params.C),' -p',{' '},num2str(m.params.eps));
    case 'nu'
        m_str = strcat('-s 4 -c',{' '},num2str(m.params.C),' -n',{' '},num2str(m.params.nu));
    otherwise 
        error('Unknown SVR method');
end
ctrl_str = strcat(ctrl_str,m_str);
switch k.name 
    case 'linear'
       k_str = ' -t 0';
    case 'polynomial'
        k_str = strcat(' -t 1 -d',{' '},num2str(k.params.degree));
    case 'rbf'
         k_str = strcat(' -t 2 -g',{' '},num2str(k.params.width));
    otherwise 
        error('Unknown SVR method');
end
ctrl_str = char(strcat(ctrl_str,k_str,' -q'));

% run
model = svmtrain(targets,inputs,ctrl_str);

% plot
if (f)
    x = (xmin:0.1:xmax)';
    y = true_f(x);
    label = svmpredict(x,x,model,'-q');
    figure
    hold on;
    grid minor;
    set(gca, 'FontSize', 14);
    
    % plot support vectors
    p1 = scatter(inputs(model.sv_indices), targets(model.sv_indices),80*ones(size(inputs((model.sv_indices)))),'o','MarkerEdgeColor', 'g', 'MarkerFaceColor', 'w','LineWidth',1.5);
    scatter(inputs(model.sv_indices), targets(model.sv_indices),30*ones(size(inputs((model.sv_indices)))),'MarkerFaceColor','r','MarkerFaceAlpha',0.8,'MarkerEdgeColor','r','MarkerEdgeAlpha',0.8);
    % plot dataset
    p2 = scatter(inputs, targets,30*ones(size(inputs)),'MarkerFaceColor','r','MarkerFaceAlpha',0.4,'MarkerEdgeColor','r','MarkerEdgeAlpha',0.4);
    % plot reference function
    p3 = plot(x,y, 'k', 'LineWidth', 2);
    % plot prediction
    p4 = plot(x, label, '-.','Color',[0.3 0.6 1], 'LineWidth',2.2);
    
    
    xlabel('Input')
    ylabel('Output')
    legend([p1,p2,p3,p4],{'Support vectors', 'Datapoints','Target function', 'Modeled function'});
    title(strcat('Result of SVR for dataset ',{' '}, ds.name))
end

end