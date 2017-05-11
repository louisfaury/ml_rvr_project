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

[mse, BIC, trained_models] = cross_validate(Dataset, models, nfold, ttratio, 0);

display_grid_search_rvr(sigma, mse, BIC, nfold);

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
    BIC = zeros(length(sigma), length(nu), length(C), nfold);
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
            [modelMSE, modelBIC, trained_models] = cross_validate(Dataset, models, nfold, ttratio, 0);
            mse(i,j,:,:) = modelMSE';
            BIC(i,j,:,:) = modelBIC';
        end
    end
    
    display_grid_search_nusvr(Dataset, kernelstr, nfold,  nu, C, sigma, mse, BIC)
    
end

function grid_search_csvr(Dataset, ttratio, nfold, kernelstr, epsilon, C, sigma)

    mse = zeros(length(sigma), length(epsilon), length(C), nfold);
    BIC = zeros(length(sigma), length(epsilon), length(C), nfold);
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
            [modelMSE, modelBIC, trained_models] = cross_validate(Dataset, models, nfold, ttratio, 0);
            mse(i,j,:,:) = modelMSE';
            BIC(i,j,:,:) = modelBIC';
        end
    end
    
    display_grid_search_csvr(Dataset, kernelstr, nfold, epsilon, C, sigma, mse, BIC)
end

function display_grid_search_rvr(sigma, mse, BIC, nfold)


%% Display MSE
figure
hold on;
grid minor;
set(gca, 'FontSize', 12);
boxplot(mse, 'labels', roundn(sigma, -2));
xlabel('$\sigma$')
ylabel('MSE')
title(strcat(num2str(nfold), '-fold cross validation results for RVR (MSE)'));

%% Display BIC
figure
hold on;
grid minor;
set(gca, 'FontSize', 12);
boxplot(BIC, 'labels', roundn(sigma, -2));
xlabel('$\sigma$')
ylabel('BIC')
title(strcat(num2str(nfold), '-fold cross validation results for RVR (BIC)'));

end

function display_grid_search_nusvr(Dataset, kernelstr, nfold, nu, C, sigma, mse, BIC)

%% Plot MSE
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
    set(gca,'yscale','log', 'FontSize', 14)
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
    set(gca,'xscale','log', 'FontSize', 14)
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
    set(gca,'yscale','log', 'xscale', 'log', 'FontSize', 14)
    title(strcat('Grid search for $\nu$-SVR on $\sigma$ and C ($\nu$ = ', num2str(bestNu), ')'));
    
    % Plot slices
    subplot(2,2,4)
    [nuPlot, sigmaPlot, cPlot] = meshgrid(nu, sigma, C);
    slice(nuPlot, sigmaPlot, cPlot, meanMse,  [], [],  [C(1)  C(round((idx3+1)/2)) C(idx3)], 'linear')
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\nu$');
    ylabel('$\sigma$');
    zlabel('$C$');
    zlim([C(1), C(idx3)]);
    set(gca,'yscale','log', 'zscale', 'log', 'FontSize', 14)
    title(strcat('Grid search for $\nu$-SVR'));
    
    
%% Plot BIC
    meanBIC = mean(BIC, 4);
    [idx1, idx2, idx3] = ind2sub(size(meanBIC),find(meanBIC == min(meanBIC(:))));
    bestSigma = sigma(idx1);
    bestNu    = nu(idx2);
    bestC     = C(idx3);
    
    
    figure
    % Plot grid search nu against C
    subplot(2,2,1)
    [cPlot, nuPlot] = meshgrid(C, nu);
    pcolor(nuPlot, cPlot, squeeze(meanBIC(idx1,:,:)))
    set(gca,'yscale','log', 'FontSize', 14)
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold BIC'));
    xlabel('$\nu$');
    ylabel('C')
    title(strcat('Grid search for $\nu$-SVR on C and $\nu$ ($\sigma$ = ', num2str(bestSigma), ')'));
    
   
    % Plot grid search sigma against nu
    subplot(2,2,2)
    [nuPlot, sigmaPlot] = meshgrid(nu, sigma);
    pcolor(sigmaPlot, nuPlot, squeeze(meanBIC(:,:,idx3)))
    shading interp
    clb = colorbar;
    set(gca,'xscale','log', 'FontSize', 14)
    title(clb, strcat(num2str(nfold), '-fold BIC'));
    xlabel('$\sigma$');
    ylabel('$\nu$')
    title(strcat('Grid search for $\nu$-SVR on $\sigma$ and $\nu$ (C = ', num2str(bestC), ')'));
    
    % Plot grid search sigma against C
    subplot(2,2,3)
    [cPlot, sigmaPlot] = meshgrid(C, sigma);
    pcolor(sigmaPlot, cPlot, squeeze(meanBIC(:,idx2,:)))
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold BIC'));
    xlabel('$\sigma$');
    ylabel('$C$')
    set(gca,'yscale','log', 'xscale', 'log', 'FontSize', 14)
    title(strcat('Grid search for $\nu$-SVR on $\sigma$ and C ($\nu$ = ', num2str(bestNu), ')'));
    
    % Plot slices (log scale colormap)
    subplot(2,2,4)
    [nuPlot, sigmaPlot, cPlot] = meshgrid(nu, sigma, C);
    
    d = log(meanBIC);
    slice(nuPlot, sigmaPlot, cPlot, meanBIC,  [], [],  [C(1)  C(round((idx3+1)/2)) C(idx3)], 'linear')
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold BIC'));
    xlabel('$\nu$');
    ylabel('$\sigma$');
    zlabel('$C$');
    zlim([C(1), C(idx3)]);
    ylim([sigma(1), sigma(end)])
    xlim([nu(1), nu(end)])
    set(gca,'yscale','log', 'zscale', 'log', 'FontSize', 14)
    title(strcat('Grid search for $\nu$-SVR'));



    % Plot best model
    kernel = generate_kernel(kernelstr, bestSigma);
    name = strcat('$\nu$-SVR $\sigma$=', num2str(bestSigma), '$\nu$=',  num2str(bestNu), 'C =', num2str(bestC));
    model  = generate_SVR('nu', kernel, bestC, bestNu, name);
    model = train_model(Dataset,model,1);
    

end

function display_grid_search_csvr(Dataset, kernelstr, nfold, epsilon, C, sigma, mse, BIC)


%% Plot MSE
    meanMse = mean(mse, 4);
    [idx1, idx2, idx3] = ind2sub(size(meanMse),find(meanMse == min(meanMse(:))));
    bestSigma = sigma(idx1);
    bestepsilon    = epsilon(idx2);
    bestC     = C(idx3);
    
    
    figure
    % Plot grid search nu against C
    subplot(2,2,1)
    [cPlot, epsPlot] = meshgrid(C, epsilon);
    pcolor(epsPlot, cPlot, squeeze(meanMse(idx1,:,:)))
    set(gca,'yscale','log', 'xscale', 'log', 'FontSize', 14)
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\epsilon$');
    ylabel('C')
    title(strcat('Grid search for $\epsilon$-SVR on C and $\epsilon$ ($\sigma$ = ', num2str(bestSigma), ')'));
    
   
    % Plot grid search sigma against nu
    subplot(2,2,2)
    [epsPlot, sigmaPlot] = meshgrid(epsilon, sigma);
    pcolor(sigmaPlot, epsPlot, squeeze(meanMse(:,:,idx3)))
    set(gca,'yscale','log', 'xscale', 'log', 'FontSize', 14)
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\sigma$');
    ylabel('$\epsilon$')
    title(strcat('Grid search for $\epsilon$-SVR on $\sigma$ and $\epsilon$ (C = ', num2str(bestC), ')'));
    
    % Plot grid search sigma against C
    subplot(2,2,3)
    [cPlot, sigmaPlot] = meshgrid(C, sigma);
    pcolor(sigmaPlot, cPlot, squeeze(meanMse(:,idx2,:)))
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\sigma$');
    ylabel('$C$')
    set(gca,'yscale','log', 'xscale', 'log', 'FontSize', 14)
    title(strcat('Grid search for $\epsilon$-SVR on $\sigma$ and C ($\epsilon$ = ', num2str(bestepsilon), ')'));
 
    % Plot slices 
    subplot(2,2,4)
    [epsPlot, sigmaPlot, cPlot] = meshgrid(epsilon, sigma, C);
    slice(epsPlot, sigmaPlot, cPlot, meanMse, [], [],  [C(1)  C(round((idx3+1)/2)) C(idx3)], 'linear')
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\epsilon$');
    ylabel('$\sigma$');
    zlabel('$C$');
    zlim([C(1), C(idx3)]);
    set(gca,'yscale','log', 'xscale', 'log', 'zscale', 'log', 'FontSize', 14)
    title(strcat('Grid search for $\epsilon$-SVR'));

%% Plot MSE

    meanBIC = mean(BIC, 4);
    [idx1, idx2, idx3] = ind2sub(size(meanBIC),find(meanBIC == min(meanBIC(:))));
    bestSigma = sigma(idx1(1));
    bestepsilon    = epsilon(idx2(1));
    bestC     = C(idx3(1));
    
    
    figure
    % Plot grid search nu against C
    subplot(2,2,1)
    [cPlot, epsPlot] = meshgrid(C, epsilon);
    pcolor(epsPlot, cPlot, squeeze(meanBIC(idx1(1),:,:)))
    set(gca,'yscale','log', 'xscale', 'log', 'FontSize', 14)
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold MSE'));
    xlabel('$\epsilon$');
    ylabel('C')
    title(strcat('Grid search for $\epsilon$-SVR on C and $\epsilon$ ($\sigma$ = ', num2str(bestSigma), ')'));
    
   
    % Plot grid search sigma against nu
    subplot(2,2,2)
    [epsPlot, sigmaPlot] = meshgrid(epsilon, sigma);
    pcolor(sigmaPlot, epsPlot, squeeze(meanBIC(:,:,idx3(1))))
    set(gca,'yscale','log', 'xscale', 'log', 'FontSize', 14)
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold BIC'));
    xlabel('$\sigma$');
    ylabel('$\epsilon$')
    title(strcat('Grid search for $\epsilon$-SVR on $\sigma$ and $\epsilon$ (C = ', num2str(bestC), ')'));
    
    % Plot grid search sigma against C
    subplot(2,2,3)
    [cPlot, sigmaPlot] = meshgrid(C, sigma);
    pcolor(sigmaPlot, cPlot, squeeze(meanBIC(:,idx2(1),:)))
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold BIC'));
    xlabel('$\sigma$');
    ylabel('$C$')
    set(gca,'yscale','log', 'xscale', 'log', 'FontSize', 14)
    title(strcat('Grid search for $\epsilon$-SVR on $\sigma$ and C ($\epsilon$ = ', num2str(bestepsilon), ')'));
 
    % Plot slices 
    subplot(2,2,4)
    [epsPlot, sigmaPlot, cPlot] = meshgrid(epsilon, sigma, C);
    slice(epsPlot, sigmaPlot, cPlot, meanBIC, [], [],  [C(1)  C(round((idx3(1)+1)/2)) C(idx3(1))], 'linear')
    shading interp
    clb = colorbar;
    title(clb, strcat(num2str(nfold), '-fold BIC'));
    xlabel('$\epsilon$');
    ylabel('$\sigma$');
    zlabel('$C$');
    zlim([C(1), C(idx3(1))]);
    ylim([sigma(1), sigma(end)])
    xlim([epsilon(1), epsilon(end)])
    set(gca,'yscale','log', 'xscale', 'log', 'zscale', 'log', 'FontSize', 14)
    title(strcat('Grid search for $\epsilon$-SVR'));
    
    % Plot best model
    kernel = generate_kernel(kernelstr, bestSigma);
    name = strcat('$\epsilon$-SVR $\sigma$=', num2str(bestSigma), '$\epsilon$=',  num2str(bestepsilon), 'C =', num2str(bestC));
    model  = generate_SVR('C', kernel, bestC, bestepsilon, name);
    model = train_model(Dataset,model,1);
    

end


