function u = dgstep_e(u,K,g,dt,a,s,epsilon,xtol,energy)
[Ny,Nx] = size(u);
Dxu = zeros(Ny+1,Nx+1);  % x-derivatives at i+1/2,j points
Dyu = zeros(Ny+1,Nx+1);  % y-derivatives at i,j+1/2 points

Dxu(2:end,2:end-1) = diff(u,1,2).^2;
Dyu(2:end-1,2:end) = diff(u).^2;

u = padarray(u,[1 1]);
Dxu = padarray(Dxu,[1 1],'post');
Dyu = padarray(Dyu,[1 1],'post');
for i = (1:Ny)+1
    for j = (1:Nx)+1
        u_old = u(i,j);
        Dx = [-Dxu(i,j),Dxu(i-1,j), Dxu(i,j-1)];
        Dy = [-Dyu(i,j),Dyu(i-1,j), Dyu(i,j-1)];
        i1 = i-1 > 1; i2 = i-1 < Ny; j1 = j-1 > 1; j2 = j-1 < Nx;

        E = energy_diff(u_old,u,g,i,j,Dx,Dy,epsilon,a,i1,i2,j1,j2);
        
        % Solve for u_ij'
        [u_new,~] = fzeroskeleton_e([-0.1,1.1],xtol,u_old,u,Dx,Dy,E,g(i-1,j-1),...
            i,j,i1,j1,i2,j2,Ny,Nx,dt,a,epsilon,energy);

        % Read next column and update difference approximations
        u(i,j) = u_new;
        if j1
            Dxu(i,j-1) = (u_new-u(i,j-1));
        end
        if j2
            Dxu(i,j) = (u(i,j+1)-u_new);
        end
        if i1
            Dyu(i-1,j) = (u_new - u(i-1,j));
        end
        if i2
            Dyu(i,j) = (u(i+1,j)-u_new);
        end
    end
%     i
end
% figure;
% imagesc(gradabsmid)
end