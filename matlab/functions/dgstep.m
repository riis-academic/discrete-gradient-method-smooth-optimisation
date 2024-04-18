function u = dgstep(u,K,g,dt,a,s,epsilon,xtol)
[Ny,Nx] = size(u);
Dxu = zeros(Ny,Nx+1);  % x-derivatives at i+1/2,j points
Dyu = zeros(Ny+1,Nx);  % y-derivatives at i,j+1/2 points

Dxu(:,2:end-1) = diff(u,1,2).^2;
Dyu(2:end-1,:) = diff(u).^2;

u = padarray(u,[1 1]);
Dxu = padarray(Dxu,[1 1],'post');
Dyu = padarray(Dyu,[1 1],'post');
sqrteps = realsqrt(epsilon);
for i = 1:Ny
    for j = 1:Nx  
        u_l = [u(i,j+1), u(i+2,j+1), u(i+1,j), u(i+1,j+2)];
        u_old = u(i+1,j+1);
        Dxul = [Dxu(i,j), Dxu(i+1,j), Dxu(i,j+1)];
        Dyut = [Dyu(i,j+1), Dyu(i,j), Dyu(i+1,j)];
        
        E = realsqrt(Dxul(1) + Dyut(2) + epsilon);
        if j < Nx
            E = E + realsqrt(Dxul(3) + Dyut(1) + epsilon);
        else
            E = E + sqrteps;
        end
        
        if i < Ny
            E = E + realsqrt(Dxul(2) + Dyut(3) + epsilon);
        else
            E = E + sqrteps;
        end
        E = a*E;
        
        E = E + K(i,j)/s*(u_old-g(i,j)).^s;
        
        % Solve for u_ij'
        [unew,~] = fzeroskeleton([-0.1,1.1],xtol,u_old,u_l,Dxul(2) + epsilon,...
            Dyut(1) + epsilon,E,g(i,j),(i>1),(j>1),(i<Ny),(j<Nx),dt,...
            a,epsilon,(~(i<Ny) +  ~(j<Nx))*sqrteps);

        % Read next column and update difference approximations
        u(i+1,j+1) = unew;
        if i > 1
            Dyu(i,j) = (unew - u_l(1))^2;
        end
        if i < Ny
            Dyu(i+1,j) = (unew - u_l(2))^2;
        end
        if j > 1
            Dxu(i,j) =  (unew - u_l(3))^2;
        end
        if j < Nx
            Dxu(i,j+1) = (unew - u_l(4))^2;
        end
        
    end
%     i
end
% figure;
% imagesc(gradabsmid)
end