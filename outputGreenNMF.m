%% Output of the simulations
function [Sf, Comp, Dr, Det, Wf] = outputGreenNMF(max_number_of_sources, RECON, SILL_AVG, numT, nd, xD, t0, S)
close all

x = 1:1:max_number_of_sources;

y1 = RECON; 
y2 = SILL_AVG;

createfigureNS(x, y1, y2)
[aic_values, aic_min, nopt] = AIC( RECON, SILL_AVG, numT, nd);

name1 = sprintf('Results/Solution_4det_%dsources.mat',nopt);
name2 = sprintf('Results/Solution_4det_%dsources.mat',nopt);


load(name1)
load(name2)

[Sf, Comp, Dr, Det, Wf] = CompRes(Cent,Solution,t0,numT,S,xD);  
disp('                           ')
disp(['The number of estimated by GreenNMF sources = ' num2str(nopt)]);
disp('                            ')
disp([ '       A           ' 'x            ' 'y             ' 'u           ' 'Dx           '   'Dy     ']);
disp(Solution)