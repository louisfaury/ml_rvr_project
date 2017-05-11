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

dx = 1; 
dy = 0;

figure
hold on;
grid minor;
set(gca, 'FontSize', 14);
xlim([0, max(nRelevant) + 5])
%xticks(1:nb_models)
%xticklabels(names)
scatter(nRelevant,mmse)
text(nRelevant+dx, mmse+dy, names);
xlabel('Number of support vectors')
ylabel('MSE')
title('Analysis of sparsity vs MSE')

end