# GreenNMF

Advection-diffusion equation describes the transport of particles, energy or other physical quantities in continuous 
media. An important problem in such systems is the identification of sources of contamination (or heat). Usually, the solution to this problem is based on solving complex ill-posed inverse models against available state-variable data records. However, if there are several sources, with different locations and strengths, the data records represent mixtures rather than the separate influences of the original sources. The number of these original release sources is typically unknown, which hinders the reliability of classical inverse-model analyses. To address this challenge, we present here a novel hybrid method for identification of the unknown number of release sources. 

Our method GreenNMF couples unsupervised learning based on Non-negative Matrix Factorization and inverse-analysis Green's functions method. GreenNMF synergistically performs decomposition of the recorded mixtures, finds the number of the unknown sources and uses the Green's function of advection-diffusion equation to identify their characteristics. 

GreenNFM code presented here is capable of identifying the advection velocity (the direction of the advection velocity is assumed TO BE on the axis x) and dispersivity of the medium as well as the unknown number, locations, and properties oftherelease sources. GreenNMF can be applied directly to any problem controlled by a partial-differential parabolic equation where mixtures of an unknown number of sources are measured at multiple locations.

# Requirements:

MATLAB Version (Version 9.3)
Parallel Computing Toolbox 
Optimization Toolbox  
Statistics Toolbox

# Example:

Set of two sources, monitored by four detectors.   
To run the example run example4d2s.m script.The script first generates the observational data.

At the end of the run the script prints out the parameters for each source, in the following order: the number of the sources their strength, x and y coordinates, advection velocity (along x), x and y components of dispersivity.
The optimal number of sources is determined automatically by the script by clustering and combining the Average Silhouette  with Reconstruction values and an AIC criterion.

At the end of its run the script also generates three plots:    
Average Silhouette and Reconstruction values as a function of the possible number of sources
Contribution of each determined source to the total signal at each detector.
Reconstruction of the generated mixtures of contaminant signals by the solutions obtained by GreenNMFk. 

Actual parameters of the sources in the generated data:

           source 1   source 2

# strength 

A [mg/L]     0.5	 0.7

# position 

x [km]	    -0.1 	-0.9 

y [km]      -0.2  	-0.8


Actual parameters of the medium:

# advection velocity 

u_x = 0.05 [km/year]

# dispersivity 

D_x = 0.005   [km^2/year]

D_y = 0.00125 [km^2/year]


Positions of detectors:

	D1	 D2	D3	D4
	
x	0	-0.5	0.5	0.5

y	0	-0.5	0.5    -0.5



