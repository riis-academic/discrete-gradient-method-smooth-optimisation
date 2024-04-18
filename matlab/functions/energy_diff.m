function out = energy_diff(u_new,u,g,i,j,Dx,Dy,epsilon,a,i1,i2,j1,j2, energy, flag)
global count energyVec
count = count+1;

Dx = [-(u(i,j+1)-u_new), Dx(2), (u_new-u(i,j-1))];
Dy = [-(u(i+1,j)-u_new), (u_new - u(i-1,j)), Dy(3)];

gsum = 0;

gsum = gsum + realsqrt(j2*Dx(1)^2+i2*Dy(1)^2 + epsilon);
gsum = gsum + i1*realsqrt( Dy(2)^2 + j2*Dx(2)^2 + epsilon );
gsum = gsum + j1*realsqrt( Dx(3)^2 + i2*Dy(3)^2 + epsilon );

out = a*gsum +(u_new-g(i-1,j-1))^2/2;
% 
% if exist('flag','var')
%     energyVec = [energyVec, energy + out];
% end

end