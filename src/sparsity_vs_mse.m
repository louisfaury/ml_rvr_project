function sparsity_vs_mse(ds, models, nb_folds, training_ratio,highlight_arr)
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
[mse,~, relevants] = cross_validate(ds, models, nb_folds, training_ratio, 0);
n_models = length(models);
mmse = mean(mse);
nRelevant = mean(relevants)/size(ds.inputs,1);
names = cell(1,n_models);


for i=1:n_models
   names{i} = models(i).name;
end
% Text displacement (readability)
dx = -0.01; 
dy = 0.00015;
figure;
hold on;
grid minor;
set(gca, 'FontSize', 10);
xlim([0, 0.6])

if exist('highlight_arr','var')
   a = scatter(nRelevant,mmse,50*ones(n_models,1),[0.2 0.5 1],'filled');
   b = scatter(nRelevant(highlight_arr),mmse(highlight_arr),100*ones(1,length(highlight_arr)),[1 0 0],'LineWidth',2);
   text(nRelevant(highlight_arr)-0.01+dx, mmse(highlight_arr)-0.005,{names{highlight_arr}},'FontSize',13,'FontWeight','bold');
   h = legend([a,b],{'With arbitrary (\nu,C,\sigma)','Optimal (\nu,C,\sigma)'});
   set(h,'FontSize',13);
else
    colormap winter;
    scatter(nRelevant,mmse,100*ones(n_models,1),1:n_models,'filled')
    text(nRelevant+dx, mmse+dy, names,'FontSize',13);
end
xlabel('Fraction of support vectors','FontSize',14)
ylabel('MSE','FontSize',14)
title('Analysis of sparsity vs MSE')
    
end