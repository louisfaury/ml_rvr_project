function K = compute_gram_matrix(kernel,X,Y)
    switch kernel.name
        case 'rbf'
            K = exp(-distSquared(X,Y)/(kernel.params.width^2));
            K = [K,ones(size(X,1),1)];
        otherwise 
            % TODO
    end
end
