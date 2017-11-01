function [RECON, SILL_AVG] = GreenNMF(max_number_of_sources,nd,Nsim,aa,xD,t0,time,S,numT)

    number_of_sources = 1;
    disp(['trying number_of_sources = ' num2str(number_of_sources)]);
    
    [sol,normF,lb,ub, AA, sol_real,normF_real, normF1, sol_all] = simulations(number_of_sources,nd,Nsim,aa,xD,t0,time,S,numT);
    %disp(size(sol));
    yy        = quantile(normF,.25);
    reconstr1 = mean(normF(normF<yy));
    
    ind  = find(normF < yy);
    sol1 = sol(ind,:);
    
    avg_sol             = mean(sol1);
    Solution            = avg_sol;
    mean_savg           = 1;
    number_of_clust_sim = 0;

    file_name1 = sprintf('./Results/Solution_%ddet_%dsources.mat',nd, number_of_sources);
    save(file_name1, 'Solution', 'reconstr1', 'mean_savg','number_of_clust_sim');  

RECON(1)   = reconstr1;
SILL_AVG(1) = 1;


number_of_sources = 2;

for jj = 2:max_number_of_sources
  
disp(['trying number_of_sources = ' num2str(number_of_sources)]);

[sol,normF,lb,ub, AA,sol_real, normF_real, normF1, sol_all, normF_abs, Qyes] = simulations(number_of_sources,nd,Nsim,aa,xD,t0,time,S,numT);


[Solution, VectIndex, Cent, reconstr, mean_savg, number_of_clust_sim] = clustering_the_solutions(number_of_sources,nd,sol_real,normF_real, Qyes);

RECON(number_of_sources)    = reconstr;
SILL_AVG(number_of_sources) = mean_savg;

number_of_sources = number_of_sources + 1;
end

close all
RECON = RECON/Nsim; 

end