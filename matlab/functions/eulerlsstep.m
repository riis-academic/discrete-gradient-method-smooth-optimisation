function u = eulerlsstep(u,g,dt,a,epsilon,energy)
global count
[Ny,Nx] = size(u);
Dxu = zeros(Ny+1,Nx+1);  % x-derivatives at i+1/2,j points
Dyu = zeros(Ny+1,Nx+1);  % y-derivatives at i,j+1/2 points

Dxu(2:end,2:end-1) = diff(u,1,2);
Dyu(2:end-1,2:end) = diff(u);
c1 = 1/2;
p = 1/2;

u = padarray(u,[1 1]);
Dxu = padarray(Dxu,[1 1],'post');
Dyu = padarray(Dyu,[1 1],'post');
% sqrteps = realsqrt(epsilon);
count_start = count;
for i = (1:Ny)+1
%     if mod(i,5) == 0
%         disp([ '[i: ' num2str(round(i/Ny*100)) '%.]'])
%     end
    for j = (1:Nx)+1
        u_old = u(i,j);
        Dx = [-Dxu(i,j), Dxu(i-1,j), Dxu(i,j-1)];
        Dy = [-Dyu(i,j), Dyu(i-1,j), Dyu(i,j-1)];
        i1 = i-1 > 1; i2 = i-1 < Ny; j1 = j-1 > 1; j2 = j-1 < Nx; 
        
        df = tv_step_new(u_old,g(i-1,j-1),Dx,Dy,i1,i2,j1,j2,epsilon,a);
        u_new = u_old - dt*df;
        f0 = energy_diff(u_old,u,g,i,j,Dx,Dy,epsilon,a,i1,i2,j1,j2);
        f1 = energy_diff(u_new,u,g,i,j,Dx,Dy,epsilon,a,i1,i2,j1,j2,energy-f0,true);
        while f1-f0 > c1*df*(u_new-u_old) && abs(u_new-u_old) > 1e-10
            u_new = u_old - p*(u_old-u_new);
            f1 = energy_diff(u_new,u,g,i,j,Dx,Dy,epsilon,a,i1,i2,j1,j2,energy-f0,true);
        end
            
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
count_end = count;
count_num = round((count_end - count_start)/Nx/Ny);
disp(count_num)
end