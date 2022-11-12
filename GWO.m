% Grey Wolf Optimizer
function [a1,Alpha_score,Alpha_pos,Convergence_curve]=GWO(n,Max_iter,lb,ub,dim,fobj,Positions)
a1=[];
Convergence_curve=zeros(1,Max_iter);
t=0; %Loop counter
% Main loop
while t<Max_iter
    %% 
    for i=1:size(Positions,1)  
        %Return back  that go beyond the boundaries of the search space
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               
        % Calculate objective function for each position
        fit_value=fobj(Positions(i,:));
        fitness1(i,1)=fit_value;
    end
    %%
    [fitness, ind]=sort(fitness1,'ascend');
    Alpha_pos=Positions(ind(1),:);
    Alpha_score=fitness(1); 
    Beta_pos=Positions(ind(2),:);
    Beta_score=fitness(2); 
    Delta_pos=Positions(ind(3),:);
    Delta_score=fitness(3); 
    %% Update the Position of the Grey Wolf
    a=2-t*((2)/Max_iter); % a decreases linearly fron 2 to 0
    a1(t+1,1)=a;
    % Update the Position of grey wolf including omegas
    for i=1:size(Positions,1)
       for j=1:size(Positions,2)
            %%           
            r1=rand(); % r1 is a random number between 0 and 1
            r2=rand(); % r2 is a random number between 0 and 1 
            
            A1=2*a*r1-a; 
            C1=2*r2; % 
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); 
            X1=Alpha_pos(j)-A1*D_alpha; 
            %%           
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; 
            C2=2*r2; 
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); 
            X2=Beta_pos(j)-A2*D_beta;        
            %%
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; 
            C3=2*r2; 
            %%
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); 
            X3=Delta_pos(j)-A3*D_delta;              
            %%
            Positions(i,j)=(X1+X2+X3)/3;
            
        end
    end
    t=t+1;    
    Convergence_curve(t)=Alpha_score;
end