function [S,XF] = observational_matrix(As,Xs,xD,D,t0,u,numT,noise,time)
%% Calculation of the observation MATRIX
nd   = size(xD,1); % the number of the detectors
S    = zeros(nd, nd*numT);% The observation matrix S (with unknown number of sources)

%% Here we are ordering the signals one after the other in one long vector 
% xxxxxxxxxxxxxxxxxxxxxxx Construction of the mixes at the detectors xxxxxxxxxxxxx
for  d = 1:nd % loop on detectors
    
    if  length(As) == 1
          Mix    = source(time, Xs(1,:), xD(d,:), D, t0, u) + noise*randn(size(time));  
          S(d,:) = [zeros(1, (d-1)*numT)  Mix  zeros(1, (nd-d)*numT)] ;
    else    
          for i = 1:length(As);%loop on the sources
                if i == 1 
                   Mix = source(time, Xs(i,:), xD(d,:), D, t0, u) + noise*randn(size(time));  
                else
                   Mix = Mix + source(time, Xs(i,:), xD(d,:), D, t0, u);
                end
          end
          S(d,:) = [zeros(1, (d-1)*numT)  Mix  zeros(1, (nd-d)*numT)] ;   
    end
end


%                        ---80---    ---80---  ---80---   --80--     --80---
%                           1           2         3         4          5
% The structure of S is [first mix   0,0...0,   0,0...0,  0,0...0    0,0...0]
%                       [0, 0...0   second mix  0,0...0,  0,0...0    0,0...0]
%                       [0, 0...0    0,0...0   third mix  0,0...0    0,0...0]
%                       [0, 0...0    0,0...0    0,0...0, fourth mix  0,0...0]
%                       [0, 0...0    0,0...0    0,0...0,  0,0...0  fifth mix]
size(S);

XF = reshape(S',1, size(S,2)*nd);% This is the long vector (its length is = nd*numT
%%
xtrue = reshape(Xs' ,1, size(Xs,2)*size(Xs,1));%The original source constants Ai Xi Yi

file_name1 = sprintf('./Results/xtrue_%ddet_%dsources.mat',nd, length(As));
save(file_name1, 'XF', 'xtrue', 'S', 'xD','D','u'); 

end