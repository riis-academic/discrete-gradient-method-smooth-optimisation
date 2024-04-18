function out = itohabe(u,u_old,u_l,Dxul,Dyut,E,g,ig1,jg1,Nygi,Nxgj,dt,a,epsilon,sqrteps)
global count
count = count+1;

s = 2;

gsum = sqrteps;
gsum = gsum + Nygi*realsqrt(Dxul  + (u - u_l(2))^2);
gsum = gsum + Nxgj*realsqrt((u - u_l(4))^2  + Dyut);
gsum = gsum + realsqrt(jg1*(u - u_l(3))^2  + ig1*(u - u_l(1))^2 + epsilon);

out = dt*(a*gsum + (u-g)^s/s - E)/(u-u_old) + u - u_old;
end