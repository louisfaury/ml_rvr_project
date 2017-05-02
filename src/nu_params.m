function type = nu_params(C,nu);
    params = struct('C',C,'nu',nu);
    type = struct('type','nu','params',params);
end