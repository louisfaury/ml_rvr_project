function models = generate_inrange_model(type,kernelparam_min,kernelparam_max,typeparam_1_min,typeparam_1_max,typeparam_2_min,typeparam_2_max)
% ============= HEADER ============= %
% \brief   - Generates an array of random models given bounding hyperparameters
%               values (so far with SVR and rbf kernels)
% \param   - type <- method type
%          - kernelparam  <- min and max for kernel params (i.e width)
%          - typeparam  <- min and max for different values of the method
% ============= HEADER ============= %

gc          = 7; % grid coarseness
%kernelparam = kernelparam_min + (kernelparam_max-kernelparam_min)*rand(1,gc);
%typeparam_1 = typeparam_1_min + (typeparam_1_max-typeparam_2_min)*rand(1,gc);
%typeparam_2 = typeparam_2_min + (typeparam_2_min-typeparam_2_min)*rand(1,gc);
kernelparam = linspace(kernelparam_min,kernelparam_max,gc);
typeparam_1 = linspace(typeparam_1_min,typeparam_1_max,gc);
typeparam_2 = linspace(typeparam_2_min,typeparam_2_max,gc);


models = [];
for i=1:gc
    kernel = generate_kernel('rbf',kernelparam(i));
    for j=1:gc
        for k=1:gc
            models = [models generate_SVR(type,kernel,typeparam_1(j),typeparam_2(k),'')];
        end
    end
end

end

%% TODO : same for RVR (if needed)