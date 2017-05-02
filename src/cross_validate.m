function mse = cross_validate(ds, models, nb_folds, training_ratio);
% ============= HEADER ============= %
% \brief   ? Performs cross validation and plots the result for all models
% \param   ? ds <- dataset 
%          ? models  <- sequence of 
%                       model { 
%                           'name' <- name for the plot
%                           'type' <- 'SVR' or 'RVR'
%                           'kernel' { 
%                                   'name'
%                                   'params'
%                                   } 
%                           'params' <- depend on the type
%                       }
%          ? nb_folds  <- number of folds for CV
%          ? training_ratio  <- ratio of training examples

% \returns ? the mse matrix for each model on each fold
% ============= HEADER ============= %

    nb_models = size(models,2);
    mse = zeros(nb_folds,nb_models);
    
    for j=1:nb_folds
        [train_fold, test_fold] = generate_fold(ds,training_ratio);
        for i=1:nb_models
           switch models(i).type
                case 'SVR'
                    model = svr(train_fold, models(i).kernel, models(i).params,false);
                    label = svmpredict(test_fold.inputs,test_fold.inputs,model,'-q');
                    
                case 'RVR'
                    model = rvr(train_fold, models(i).kernel, models(i).params,false);
                    label = model.predict(test_fold.inputs)';
                    
                otherwise 
                    error('Unknown method');
           end
           mse(j,i) = mean((test_fold.outputs - label).^2);
        end       
    end

    names = cell(1,nb_models);
    for i=1:nb_models
           names{i} = models(i).name;
    end
    
    figure
    hold on;
    grid minor;
    set(gca, 'FontSize', 14);
    xlim([0,nb_models + 1])
    xticks(1:nb_models)
    xticklabels(names)
    errorbar(1:nb_models,mean(mse),std(mse),'o')
    xlabel('Models')
    ylabel('MSE')
    title('Cross Validation results')
    
end