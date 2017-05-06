function k = generate_kernel(k_str,param)
    switch k_str
        case 'rbf'
            k = rbf_kernel(param);
        case 'linear'
            k = linear_kernel();
        case 'polynomial'
            k = polynomial_kernel(param);
        otherwise
            error('unknown kernel type')
    end
end

function k = rbf_kernel(width)
    params = struct('width',width);
    k = struct('name','rbf','params',params);
end

function k = linear_kernel()
    k = struct('name','linear','params',[]);
end

function k = polynomial_kernel(degree)
    params = struct('degree',degree);
    k = struct('name','polynomial','params',params);
end

