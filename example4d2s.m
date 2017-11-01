%% Example of the identification of two point-like release-sources in an infinite medium.
%  when the number of the sources, their parameters and the properties of the 
%  medium are unknow, via the hibrid method GreenNMF 
%  that use the Green function of the advection-difusion equation in an infinite medium. 
%
% Boian S. Alexandrov, Velimir V. Vesselinov, Valentin Stanev, and Filip Iliev
% Los Alamos National Laboratory
% Los Alamos, NM, USA
% boian@lanl.gov
%
% This software and its documentation are copyright 2017 by the
% Los Alamos National Laboratory. All rights are reserved.
% This software is supplied without any warranty or guaranteed support whatsoever. 
% Neither Los Alamos National Laboratory nor the authors  
% are responsible for its use, misuse, or functionality.


%% 

clear all
close all


delete(gcp)
parpool('local');
myCluster = parcluster('local');

clc
tic

number_of_sources = 1;
generation_the_initial_setup;


[RECON, SILL_AVG]       = GreenNMF(max_number_of_sources,nd,Nsim,aa,xD,t0,time,S,numT);

[Sf, Comp, Dr, Det, Wf] = outputGreenNMF(max_number_of_sources, RECON, SILL_AVG, numT, nd, xD, t0, S);

toc