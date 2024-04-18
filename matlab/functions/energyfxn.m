function E = energyfxn(u,g,a,epsilon)
s=2;
[Ny,Nx] = size(u);
Dxu = zeros(Ny,Nx+1);  % x-derivatives at i+1/2,j points
Dyu = zeros(Ny+1,Nx);  % y-derivatives at i,j+1/2 points

Dxu(:,2:end-1) = diff(u,1,2);
Dyu(2:end-1,:) = diff(u);

gradabsmid = realsqrt(Dxu(:,2:end).^2 + Dyu(2:end,:).^2 + epsilon);

E = a*sum(sum(gradabsmid)) + 1/s*sum(sum((abs(u-g)).^s));
end