function [S,XF, W] = observation_generation(As,Xs,xD,D,t0,u,numT,time)
%% Calculation of the observation MATRIX
nd   = size(xD,1); % the enumber of the detectors
S    = zeros(nd, nd*numT);% The observation matrix S (with unknown number of sources)
W    = zeros(nd,length(As));


%% Here we are ordering the signals one after the other in one long vector 
% xxxxxxxxxxxxxxxxxxxxxxx Construction of the mixes at the detectors xxxxxxxxxxxxx
for  i = 1:nd % loop on detectors
    
    for d = 1:length(As);%loop on the sources
        if d == 1 
            Mix = source(time, Xs(d,:), xD(i,:), D, t0, u);% + noise*randn(size(time));
            W(i,d) = sum(Mix);  
        else
          	Mix = Mix + source(time, Xs(d,:), xD(i,:), D, t0, u);
          	W(i,d) = sum(Mix);
        end
    end

    S(i,:) = [zeros(1, (i-1)*numT)  Mix  zeros(1, (nd-i)*numT)] ;
    
end
%                        ---80---    ---80---  ---80---   --80--     --80---
%                           1           2         3         4          5
% The structure of S is [first mix   0,0...0,   0,0...0,  0,0...0    0,0...0]
%                       [0, 0...0   second mix  0,0...0,  0,0...0    0,0...0]
%                       [0, 0...0    0,0...0   third mix  0,0...0    0,0...0]
%                       [0, 0...0    0,0...0    0,0...0, fourth mix  0,0...0]
%                       [0, 0...0    0,0...0    0,0...0,  0,0...0  fifth mix]
size(S);

Wo = W;
for j = 2:size(W,2)
    Wo(:,j) = W(:,j)-W(:,j-1);
end
W = Wo;

for i = 1:size(W,1)
    W(i,:) = W(i,:)./sum(W(i,:));
end


XF = reshape(S',1, size(S,2)*nd);% This is the long vector (its length is = nd*numT
%%
xtrue = reshape(Xs' ,1, size(Xs,2)*size(Xs,1));%The original source constants Ai Xi Yi

%file_name1 = sprintf('./Results/xtrue_%ddet_%dsources.mat',nd, length(As));
%save(file_name1, 'XF', 'xtrue', 'S'); 

end