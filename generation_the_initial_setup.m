%% Initial setup generation


max_number_of_sources = 4;
Nsim                  = 100;
% The # of the time points
numT = 80; 
time = linspace(0, 20, numT);

% The advection velocity is [0.05 0] in [km/year]
u     = 0.05;% [km/year]; 
% Dispersivity_long = 0.1;%[km]
% Dispersivity_long = 0.025;%[km]
D     = [0.005  0.00125] ; 

% the initial time the sources begin the release
t0    = -10; 
% the noise in the system
noise = 1*10^(-4); %the noise's strenght

% The amplitudes of the original sources
As    = [0.5 0.7];

% The number of the original sources
ns    = length(As);

% Xn -> the coordinates of the real sources
Xn    = zeros(2,length(As));
 
% The coordinates of the sources
Xn    = [[-0.1; -0.2] [-0.9;   -0.8]];
Xs    = ones(length(As), 3);

% Ordering the matrix of the sources: [A X Y]

for k = 1:size(Xs,1)
Xs(k,:) = [As(k) Xn(1,k) Xn(2,k)];
end
% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
% xD -> the positions of the detectors
xd1 =  [0  0];      % position of the first detector
xd2 =  [-0.5 -0.5]; % position of the second detector
xd4 =  [0.5   0.5]; % position of the third detector
xd5 =  [0.5  -0.5]; % position of the fourth detector

% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

xD = [xd1;xd2;xd4;xd5];

% The number of the detectors
nd = length(xD);

aa = 1;% the length of the interval for random IC
% xxxxxxxxxxxxxxxxx Generation of the observational matrix xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

[S,XF] = observational_matrix(As,Xs,xD,D,t0,u,numT,noise,time);