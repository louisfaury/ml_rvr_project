function grid_search_cv(Dataset, regression, ttratio, nfold, kernelstr, varargin)

switch regression
    case 'SVR'
        type    = varargin{1};
        params  = varargin(2:end);
        grid_search_svr(Dataset, ttratio, nfold, type, kernelstr, params);
    case 'RVR'
        sigma = varargin{1};
        grid_search_cv_rvr(Dataset, ttratio, nfold, kernelstr, sigma);
    otherwise 
        error('Unknwn regression algorithm');
end

end

function grid_search_cv_rvr(Dataset, ttratio, nfold, kernelstr, sigma)

models = [];
for i=1:length(sigma)
   kernel = generate_kernel(kernelstr, sigma(i));
   models = [models generate_RVR(kernel, 0.1, 0.1, strcat('RVR \sigma=', num2str(sigma(i))))];
end

mse = cross_validate(Dataset, models, nfold, ttratio, 0);

display_grid_search_rvr(sigma, mse, nfold);

end

function grid_search_svr(Dataset, ttratio, nfold, type, kernelstr, params)

switch type
    case 'nu'
        nu    = params{1};
        C     = params{2};
        sigma = params{3};
        grid_search_nusvr(Dataset, ttratio, nfold, kernelstr, nu, C, sigma);
    case 'C'
        epsilon   = params{1};
        C     = params{2};
        sigma = params{3};
        grid_search_csvr(Dataset, ttratio, nfold, kernelstr, epsilon, C, sigma);
    otherwise
        error('Unknown type of SVR')
end

end

function grid_search_nusvr(Dataset, ttratio, nfold, kernelstr, nu, C, sigma)

    mse = zeros(length(sigma), length(nu), length(C), nfold);
    for i=1:length(sigma)
        disp(i)
        for j=1:length(nu)
            disp(j)
            models = [];
            for k=1:length(C)
                kernel = generate_kernel(kernelstr, sigma(i));
                name = strcat('$\nu$-SVR $\sigma$=', num2str(sigma(i)), '\nu=',  num2str(nu(j)), 'C =', num2str(C(k)));
                models = [models generate_SVR('nu', kernel, C(k), nu(j), name)];
            end
            mse(i,j,:,:) = (cross_validate(Dataset, models, nfold, ttratio, 0))';
        end
    end
    
    display_grid_search_nusvr(Dataset, kernelstr, nfold,  nu, C, sigma, mse)
    
end

function grid_search_csvr(Dataset, ttratio, nfold, kernelstr, epsilon, C, sigma)

    mse = zeros(length(sigma), length(epsilon), length(C), nfold);
    for i=1:length(sigma)
        disp(i)
        for j=1:length(epsilon)
            disp(j)
            models = [];
            for k=1:length(C)
                kernel = generate_kernel(kernelstr, sigma(i));
                name = strcat('$\C$-SVR $\sigma$=', num2str(sigma(i)), '\epsilon=',  num2str(epsilon(j)), 'C =', num2str(C(k)));
                models = [models generate_SVR('C', kernel, C(k), epsilon(j), name)];
            end
            mse(i,j,:,:) = (cross_validate(Dataset, models, nfold, ttratio, 0))';
        end
    end
    
    display_grid_search_csvr(Dataset, kernelstr, nfold, epsilon, C, sigma, mse)
end

function display_grid_search_rvr(sigma, mse, nfold)

figure
hold on;
grid minor;
set(gca, 'FontSize', 12);
boxplot(mse, 'labels', roundn(sigma, -2));
xlabel('$\sigma$')
ylabel('MSE')
title(strcat(num2str(nfold), '-fold cross validation results for RVR'));

%TODO add plots for other metrics

end

function display_grid_search_nusvr(Dataset, kernelstr, nfold, nu, C, sigma, mse)

    meanMse = mean(mse, 4);
    [idx1, idx2, idx3] = ind2sub(size(meanMse),find(meanMse == min(meanMse(:))));
    bestSigma = sigma(idx1);
    bestNu    = nu(idx2);
    bestC     = C(idx3);
    
    
    figure
    % Plot grid search nu against C
    subplot(2,2,1)
    [cPlot, nuPlot] = meshgrid(C, nu);
    pcolor(nuPlot, cPlot, squeeze(meanMse(idx1,:,:)))
    set(gca,'yscale','log')
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\nu$');
    ylabel('C')
    title(strcat('Grid search for $\nu$-SVR on C and $\nu$ ($\sigma$ = ', num2str(bestSigma), ')'));
    
   
    % Plot grid search sigma against nu
    subplot(2,2,2)
    [nuPlot, sigmaPlot] = meshgrid(nu, sigma);
    pcolor(sigmaPlot, nuPlot, squeeze(meanMse(:,:,idx3)))
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\sigma$');
    ylabel('$\nu$')
    title(strcat('Grid search for $\nu$-SVR on $\sigma$ and $\nu$ (C = ', num2str(bestC), ')'));
    
    % Plot grid search sigma against C
    subplot(2,2,3)
    [cPlot, sigmaPlot] = meshgrid(C, sigma);
    pcolor(sigmaPlot, cPlot, squeeze(meanMse(:,idx2,:)))
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\sigma$');
    ylabel('$C$')
    set(gca,'yscale','log')
    title(strcat('Grid search for $\nu$-SVR on $\sigma$ and C ($\nu$ = ', num2str(bestNu), ')'));
    
    % Plot best model
    kernel = generate_kernel(kernelstr, bestSigma);
    name = strcat('$\nu$-SVR $\sigma$=', num2str(bestSigma), '$\nu$=',  num2str(bestNu), 'C =', num2str(bestC));
    model  = generate_SVR('nu', kernel, bestC, bestNu, name);
    model = train_model(Dataset,model,1);
    
    % TODO: add fourth plot for other metrics
    
    
    

end

function display_grid_search_csvr(Dataset, kernelstr, nfold, epsilon, C, sigma, mse)

    meanMse = mean(mse, 4);
    [idx1, idx2, idx3] = ind2sub(size(meanMse),find(meanMse == min(meanMse(:))));
    bestSigma = sigma(idx1);
    bestepsilon    = epsilon(idx2);
    bestC     = C(idx3);
    
    
    figure
    % Plot grid search nu against C
    subplot(2,2,1)
    [cPlot, nuPlot] = meshgrid(C, epsilon);
    pcolor(nuPlot, cPlot, squeeze(meanMse(idx1,:,:)))
    set(gca,'yscale','log')
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\epsilon$');
    ylabel('C')
    title(strcat('Grid search for C-SVR on C and $\epsilon$ ($\sigma$ = ', num2str(bestSigma), ')'));
    
   
    % Plot grid search sigma against nu
    subplot(2,2,2)
    [nuPlot, sigmaPlot] = meshgrid(epsilon, sigma);
    pcolor(sigmaPlot, nuPlot, squeeze(meanMse(:,:,idx3)))
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\sigma$');
    ylabel('$\epsilon$')
    title(strcat('Grid search for C-SVR on $\sigma$ and $\epsilon$ (C = ', num2str(bestC), ')'));
    
    % Plot grid search sigma against C
    subplot(2,2,3)
    [cPlot, sigmaPlot] = meshgrid(C, sigma);
    pcolor(sigmaPlot, cPlot, squeeze(meanMse(:,idx2,:)))
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\sigma$');
    ylabel('$C$')
    set(gca,'yscale','log')
    title(strcat('Grid search for $\epsilon$-SVR on $\sigma$ and C ($\epsilon$ = ', num2str(bestepsilon), ')'));
    
    % Plot best model
    kernel = generate_kernel(kernelstr, bestSigma);
    name = strcat('$\epsilon$-SVR $\sigma$=', num2str(bestSigma), '$\epsilon$=',  num2str(bestepsilon), 'C =', num2str(bestC));
    model  = generate_SVR('C', kernel, bestC, bestepsilon, name);
    model = train_model(Dataset,model,1);
    
    % TODO: add fourth plot for other metrics
    
    
    

end


