function model = generate_SVR(type, kernel, C, param, name)
    switch type
        case 'nu'
            t = nu_params(C,param);
            model = struct('type', 'SVR', 'kernel', kernel, 'params', t, 'name', name);
        case 'C'
            t = c_params(C,param);
            model = struct('type', 'SVR', 'kernel', kernel, 'params', t, 'name', name);
        otherwise
            error('Unknown type')
    end
end