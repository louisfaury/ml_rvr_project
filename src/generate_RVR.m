function model = generate_RVR(kernel, alpha, beta, name);
    params = rvr_params(alpha,beta);
    model = struct('type', 'RVR', 'kernel', kernel, 'params', params, 'name', name);
end