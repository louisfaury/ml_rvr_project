function type = c_params(C,eps);
    params = struct('C',C,'eps',eps);
    type = struct('type','C','params',params);
end