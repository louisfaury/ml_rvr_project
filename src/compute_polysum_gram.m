function gram = compute_polysum_gram(inputs,params)
% ============= HEADER ============= %
% \brief   - Generates the Gram matrix (k(x,x')) for inputs  
% \param   - inputs <- dataset variate signal  
%          - params <- hp
% \returns - Gram matrix of scalar products in feature space
% ============= HEADER ============= %
n = size(inputs,1);

% adds k1(x,x') and k2(x,x')
gram = (params.r1 +  params.g1*(inputs*inputs')).^(params.d1) + (params.r2 + params.g2*(inputs*inputs')).^(params.d2);

% adds inputs serial number : 
gram = [(1:n)',gram];

end