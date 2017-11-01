function [sol,normF,lb,ub, AA,sol_real, normF_real, normF1, sol_all, normF_abs, Qyes] = simulations(number_of_sources,nd,Nsim,aa,xD,t0,time,S,numT)                                                 
%% Calculation of the soulution with j sources 

  
    sol   = zeros(Nsim,3*number_of_sources+3);% Each solution(original source) is with a structure: [Ai Xi Yi Ux Dx Dy] i=1,2,...kk;
    normF = zeros(Nsim,1);% norm F is the LSQ difference between the Observation and the reconstruction
    normCut = 0;
    Qyes=0;
    
    
  % Defining the function that we will minimize Funf = Sum_i(MixFn(i)-Sum_k(Sources(i,k)))^2      
       for i=1:nd
           
          if  number_of_sources == 1
          
              Mixfn = @(x) source(time, x(4:6), xD(i,:), x(1:2), t0, x(3));  
          
          else
                for d=1:number_of_sources
                    if d == 1 
                        Mixfn   = @(x) source(time, x(4:6), xD(i,:), x(1:2), t0, x(3)); 
                    else
                        mixfun2 = @(x) source(time, x(d*3+1:d*3+3), xD(i,:), x(1:2), t0, x(3));
                        Mixfn   = @(x) Mixfn(x) + mixfun2(x);
                    end
                end
          end    
                 
            if i==1  

                funF = @(x) ([Mixfn(x)  zeros(1, (nd-1)*numT)]- S(i,:));
            else
                fun2 = @(x) ([ zeros(1, (i-1)*numT) Mixfn(x)  zeros(1, (nd-i)*numT)]- S(i,:));

                funF = @(x) funF(x) + fun2(x);
            end


        end
      
     % Defining the lower and upper boundary for the minimization
       lb = [0   0     0];  % lower boundary [Ux Dx Dy]
       ub = [1   1     1];  % upper boundary [Ux Dx Dy]
     % This loop is on the number of the sources we investigate 
     % we need limits for all sources (ampl and coord)
      for jj = 1:number_of_sources;
          lb = [lb 0  -aa -aa];% General lower boundary [Ux Dx Dy A X Y]
          ub = [ub 1.5 aa  aa];% General upper boundary [Ux Dx Dy A X Y]
      end
    % The norm of the observsational matrix/vector
       AA = 0;SS = 0;for i = 1:nd;SS = S(i,:).^2;AA = AA+sum(SS);end
     
        
       
  Real_num =0;    
  sol_real =[];
  normF_real =[];
  normF1 =[];
  sol_all = [];
  j_all=0;
  DidItGoBack =0 ;
  
  cutNum = 5;
  
 while (Real_num < cutNum) & (j_all < 10*Nsim) % 
     
     
     options      = optimset('MaxFunEvals', 3000, 'display','off');
     initCON      = zeros(Nsim,3*number_of_sources+3);      
         
         for k=1:Nsim % This loop iterating the Init values
         x_init =[rand(1) 2*aa*(0.5 - rand(1, 2))];% the IC random [A X Y]
           for d = 1:number_of_sources
                 x_init = [x_init  rand() 2*aa*(0.5 - rand(1, 2))]; % the size is 3*number_of_sources+3 for the IC  
           end
         initCON(k,:) = x_init;
         end
     
     
     
    
    parfor k = 1:Nsim % This loop is iterating the NMF runs
         [sol(k,:), normF(k)] = lsqnonlin(funF,initCON(k,:),lb,ub,options);
    end
  
  normF_abs = normF;  
  normF = sqrt(normF./AA).*100;
  
  normCut = 0.1;
  
  
    index_real = find(normF < normCut);
    
  
    Real_num   = Real_num + length(index_real);
    normF_real = [normF_real; normF(index_real)];
    sol_real   = [sol_real; sol(index_real,:)];
    normF1     = [normF1; normF];
    sol_all    = [sol_all; sol];
    
    j_all = j_all+Nsim;
    
    
    if j_all == 10*Nsim & Real_num > 0 & Real_num <cutNum
         DidItGoBack = DidItGoBack+1;
        j_all =0;
    end
    
        
 end
 
 if Real_num < cutNum
     sol_real = sol_all;
     normF_real = normF1;
     Qyes = 1;
 end
     
   
        
    
            
    file_name1 = sprintf('./Results/Results_%ddet_%dsources.mat',nd, number_of_sources);
    save(file_name1, 'sol', 'normF', 'S', 'lb', 'ub', 'AA', 'sol_real', 'normF_real', 'normF_abs', 'DidItGoBack','Qyes'); 
     
    
    
 end

       