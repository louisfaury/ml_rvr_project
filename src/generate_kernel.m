function k = generate_kernel(k_str,param1,param2,param3,param4,param5,param6)
    switch k_str
        case 'rbf'
            k = rbf_kernel(param1);
        case 'linear'
            k = linear_kernel();
        case 'polynomial'
            k = polynomial_kernel(param1);
        case 'polysum'
            k = polysum_kernel(param1,param2,param3,param4,param5,param6);  % k(x,x') = (r1+g1<x,x'>)^d1 + (r2+g2<x,x'>)^d2
            % - param1 : d1 
            % - param2 : d2
            % - param3 : r1
            % - param4 : r2 
            % - param5 : g1
            % - param6 : g2 
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

function k = polysum_kernel(d1,d2,r1,r2,g1,g2)
    params = struct('d1',d1,'d2',d2,'r1',r1,'r2',r2,'g1',g1,'g2',g2);
    k = struct('name','polysum','params',params);
end

