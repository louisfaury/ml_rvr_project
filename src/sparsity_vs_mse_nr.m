function sparsity_vs_mse_nr(ds, models, nb_folds, training_ratio, is_bic)
% ============= HEADER ============= %
% \brief   - Performs cross validation and plots the result for all models,
%            as a ROC curve comparing performance (MSE) and sparsity 
% \param   - ds <- dataset
%          - models  <- sequence of
%                       model {
%                           'name' <- name for the plot
%                           'type' <- 'SVR' or 'RVR'
%                           'kernel' {
%                                   'name'
%                                   'params'
%                                   }
%                           'params' <- depend on the type
%                       }
%          - nb_folds  <- number of folds for CV
%          - training_ratio  <- ratio of training examples
%          - highlight array <- points to highlight
% ============= HEADER ============= %
[mse,~, relevants] = cross_validate(ds, models, nb_folds, training_ratio,ds.variance,0);
n_models = length(models);
mmse = mean(mse);
nRelevant = mean(relevants)/size(ds.inputs,1);
names = cell(1,n_models);
bic_indices = logical(is_bic);
mse_indices = logical(1 - is_bic);

for i=1:n_models
   names{i} = models(i).name;
end
% Text displacement (readability)
dx = - 0.01 * ones(size(nRelevant)); 
dy = - 0.0025 * ones(size(nRelevant));
dy(length(nRelevant)) = 0.0025
dx(length(nRelevant)) = - 0.04
figure;
hold on;
grid minor;
set(gca, 'FontSize', 10);
xlim([0, 0.6])
ylim([0.1, 0.19])

colors = autumn(10);
colormap(colors(1:5,:));
a = scatter(nRelevant(bic_indices),mmse(bic_indices),50*ones(sum(bic_indices),1),[1 0 0], 'filled');
b = scatter(nRelevant(mse_indices),mmse(mse_indices),50*ones(sum(mse_indices),1),[0 0 1],'filled');
text(nRelevant + dx, mmse + dy,names,'FontSize',13,'FontWeight','bold');
h = legend([a,b],{'BIC','MSE'});
set(h,'FontSize',13);

xlabel('Fraction of support vectors','FontSize',14)
ylabel('MSE','FontSize',14)
title('Analysis of sparsity vs MSE')
    
end