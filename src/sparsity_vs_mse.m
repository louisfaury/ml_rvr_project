function sparsity_vs_mse(ds, models, nb_folds, training_ratio)
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
% ============= HEADER ============= %

[mse,~, relevants] = cross_validate(ds, models, nb_folds, training_ratio, 0);
n_models = length(models);
mmse = mean(mse);
nRelevant = mean(relevants)/size(ds.inputs,1);
names = cell(1,n_models);



for i=1:n_models
   names{i} = models(i).name;
end

dx = -0.01; 
dy = 0.00015;

figure
hold on;
grid minor;
set(gca, 'FontSize', 10);
xlim([0, 0.6])
%xticks(1:nb_models)
%xticklabels(names)
colormap winter;
scatter(nRelevant,mmse,100*ones(5,1),[1 2 3 4 5],'filled')
text(nRelevant+dx, mmse+dy, names,'FontSize',13);
xlabel('Fraction of support vectors','FontSize',14)
ylabel('MSE','FontSize',14)
title('Analysis of sparsity vs MSE')

end