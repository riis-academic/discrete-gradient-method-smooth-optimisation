function out = tv_step_new(u_old,b,Dx,Dy,i1,i2,j1,j2,epsilon,a)
global count
count = count+1;


gsum = 0;

gsum = gsum + (i2*Dy(1)+j2*Dx(1))/realsqrt(j2*Dx(1)^2+i2*Dy(1)^2+epsilon);
gsum = gsum + i1*Dy(2)/realsqrt( Dy(2)^2 + j2*Dx(2)^2 + epsilon );
gsum = gsum + j1*Dx(3)/realsqrt( Dx(3)^2 + i2*Dy(3)^2 + epsilon );

out = a*gsum + u_old - b;

end