function model = train_model(ds,m, plot_flag)
   switch m.type
       case 'SVR'
           model = svr(ds,m.kernel,m.params,plot_flag);
       case 'RVR'
           model = rvr(ds,m.kernel,m.params,plot_flag);
       otherwise
           error('Unknown regression type')
   end
end