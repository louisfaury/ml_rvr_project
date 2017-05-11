function model = svr(ds,k,m,f)
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
xmin    = ds.minInput;
xmax    = ds.maxInput;
n       = ds.numPoints;

% param settings 
ctrl_str = '';
switch m.type
    case 'C'
        m_str = strcat('-s 3 -c',{' '},num2str(m.params.C),' -p',{' '},num2str(m.params.eps));
        eps = m.params.eps;
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
        k_str = strcat(' -t 1 -r 0.1 -g 0.01 -d',{' '},num2str(k.params.degree));
    case 'rbf'
         k_str = strcat(' -t 2 -g',{' '},num2str(1/(k.params.width)^2));
    otherwise 
        error('Unknown SVR method');
end
ctrl_str = char(strcat(ctrl_str,k_str,' -q'));

% run
model = svmtrain(targets,inputs,ctrl_str);


% plot
if (f)
    true_f  = str2func(ds.function);
    % nu-SVM : get equivalent epsilon 
    if (m.type == 'nu')
        tube_boundary_points_ind = model.sv_indices(find(abs(model.sv_coef)~=m.params.C)); % index of points laying on the eps-insensitive tube 
        l = svmpredict(zeros(size(tube_boundary_points_ind)),inputs(tube_boundary_points_ind),model,'-q');
        eps = mean(abs(targets(tube_boundary_points_ind)-l));
    end
    
    x = (xmin:0.1:xmax)';
    y = true_f(x);
    label = svmpredict(ones(size(x,1),1),x,model,'-q');
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
    % plot prediction and epsilon-tube
    p4 = plot(x, label, '-','Color',[0.3 0.6 1], 'LineWidth',2.2);
    p5 = plot(x,label+eps,'-.','Color',[0.3 0.6 1], 'LineWidth',1.1);
    plot(x,label-eps,'-.','Color',[0.3 0.6 1], 'LineWidth',1.1);

    
    xlabel('Input')
    ylabel('Output')
    legend([p1,p2,p3,p4,p5],{'Support vectors', 'Datapoints','Target function', 'Modeled function','Epsilon tube'});
    title(strcat('Result of SVR for dataset ',{' '}, ds.name))
end

end