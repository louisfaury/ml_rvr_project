function [train_fold, test_fold] = generate_fold(dataset, training_ratio);
% ============= HEADER ============= %
% \brief   ? Generates a fold from the original dataset 
% \param   ? ds <- original dataset 
%          ? folds_size <- number of points to keep
% \returns ? train_fold <- training instance
%          ? test_fold  <- test instance
% ============= HEADER ============= %
    n = dataset.numPoints;

    folds_size = ceil(n * training_ratio);
    fold_samples = randsample(n,folds_size);

    train_mask = zeros(1,n);
    train_mask(fold_samples) = 1;
    train_fold = generate_fold_aux(dataset, logical(train_mask),folds_size);

    test_mask = ones(1,n);
    test_mask(fold_samples) = 0;
    test_fold = generate_fold_aux(dataset, logical(test_mask),folds_size);
end

%Moche, je suis preneur si vous avez autre chose
function fold = generate_fold_aux(ds,mask,folds_size);
    fold = struct('inputs',ds.inputs(mask),'outputs',ds.outputs(mask),'function',ds.function);
    fold.referenceOutputs = ds.referenceOutputs;
    fold.name = ds.name;
    fold.numPoints = folds_size;
    fold.minInput = min(fold.outputs);
    fold.maxInput = max(fold.inputs);
end