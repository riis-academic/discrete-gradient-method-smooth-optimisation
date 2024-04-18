function u = eulerstep(u,g,dt,a,epsilon)
[Ny,Nx] = size(u);
Dxu = zeros(Ny+1,Nx+1);  % x-derivatives at i+1/2,j points
Dyu = zeros(Ny+1,Nx+1);  % y-derivatives at i,j+1/2 points

Dxu(2:end,2:end-1) = diff(u,1,2);
Dyu(2:end-1,2:end) = diff(u);

u = padarray(u,[1 1]);
Dxu = padarray(Dxu,[1 1],'post');
Dyu = padarray(Dyu,[1 1],'post');
% sqrteps = realsqrt(epsilon);
for i = (1:Ny)+1
    for j = (1:Nx)+1
        u_old = u(i,j);
        Dx = [-Dxu(i,j),Dxu(i-1,j), Dxu(i,j-1)];
        Dy = [-Dyu(i,j),Dyu(i-1,j), Dyu(i,j-1)];
        i1 = i-1 > 1; i2 = i-1 < Ny; j1 = j-1 > 1; j2 = j-1 < Nx;

        % Read next column and update difference approximations
        u_new = u_old - dt*tv_step_new(u_old,g(i-1,j-1),Dx,Dy,i1,i2,j1,j2,epsilon,a);
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
end
end