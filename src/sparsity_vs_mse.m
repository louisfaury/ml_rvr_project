function sparsity_vs_mse(ds, models, nb_folds, training_ratio);

[mse, BIC, relevants] = cross_validate(ds, models, nb_folds, training_ratio, 0);

n_models = length(models);
mmse = mean(mse);
%mBIC = mean(BIC);
nRelevant = mean(relevants);
names = cell(1,n_models);



for i=1:n_models
   names{i} = models(i).name;
end

dx = -3; 
dy = 0.00015;

figure
hold on;
grid minor;
set(gca, 'FontSize', 10);
xlim([0, max(nRelevant) + 5])
%xticks(1:nb_models)
%xticklabels(names)
colormap winter;
scatter(nRelevant,mmse,100*ones(5,1),[1 2 3 4 5],'filled')
text(nRelevant+dx, mmse+dy, names,'FontSize',13);
xlabel('Number of support vectors','FontSize',14)
ylabel('MSE','FontSize',14)
title('Analysis of sparsity vs MSE')

end