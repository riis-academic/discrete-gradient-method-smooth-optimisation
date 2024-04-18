function [b,fval] = fzeroskeleton_e(x,tol,u_old,u,Dx,Dy,E,g,i,j,i1,j1,i2,j2,Ny,Nx,dt,aa,epsilon,energy)
badint = 0;
% Interval input
if (numel(x) == 2) 
 
    a = x(1);
    b = x(2);    
    fa = itohabe_e(a,u_old,E,u,g,i,j,Dx,Dy,epsilon,aa,i1,i2,j1,j2,dt,energy);
  
    fb = itohabe_e(b,u_old,E,u,g,i,j,Dx,Dy,epsilon,aa,i1,i2,j1,j2,dt,energy);
    
%    energyfxn()
    
    if fa*fb > 0
%         disp('Wrong start interval');
        badint = 1;
        x = 0.01;
    end
    
    if ~fa
        b = a;
        return
    elseif ~fb
        % b = b;

        return
    end
end
    % Starting guess scalar input
if (numel(x) == 1) || badint

    
    fx = itohabe_e(x,u_old,E,u,g,i,j,Dx,Dy,epsilon,aa,i1,i2,j1,j2,dt,energy);

    if fx == 0
        b = x;
 
        return

    end
    
    if x ~= 0
        dx = x/5;
    else 
        dx = 1/50;
    end
    
    % Find change of sign.
    twosqrt = sqrt(2); 
    a = x; fa = fx; b = x; fb = fx;
    


    while (fa > 0) == (fb > 0)
        dx = twosqrt*dx;
        a = x - dx;  fa = itohabe_e(a,u_old,E,u,g,i,j,Dx,Dy,epsilon,aa,i1,i2,j1,j2,dt,energy);

        if ~isfinite(fa) || ~isreal(fa) || ~isfinite(a)
            b = NaN;
            return
        end

        if (fa > 0) ~= (fb > 0) % check for different sign
            % Before we exit the while loop, print out the latest interval
 
            break
        end
        
        b = x + dx;  fb = itohabe_e(b,u_old,E,u,g,i,j,Dx,Dy,epsilon,aa,i1,i2,j1,j2,dt,energy);
 
   

    end % while
end % if (numel(x) == 2)

fc = fb;
% Main loop, exit from middle of the loop
while fb ~= 0 && a ~= b
    % Insure that b is the best result so far, a is the previous
    % value of b, and c is on the opposite side of the zero from b.
    if fb*fc > 0
        c = a;  fc = fa;
        d = b - a;  e = d;
    end
    if abs(fc) < abs(fb)
        a = b;    b = c;    c = a;
        fa = fb;  fb = fc;  fc = fa;
    end
    
    % Convergence test and possible exit
    m = 0.5*(c - b);
    toler = 2.0*tol*max(abs(b),1.0);
    if (abs(m) <= toler) || (fb == 0.0) 
        break
    end
    
    % Choose bisection or interpolation
    if (abs(e) < toler) || (abs(fa) <= abs(fb))
        % Bisection
        d = m;  e = m;
    else
        % Interpolation
        s = fb/fa;
        if (a == c)
            % Linear interpolation
            p = 2.0*m*s;
            q = 1.0 - s;
        else
            % Inverse quadratic interpolation
            q = fa/fc;
            r = fb/fc;
            p = s*(2.0*m*q*(q - r) - (b - a)*(r - 1.0));
            q = (q - 1.0)*(r - 1.0)*(s - 1.0);
        end
        if p > 0, q = -q; else p = -p; end
        % Is interpolated point acceptable
        if (2.0*p < 3.0*m*q - abs(toler*q)) && (p < abs(0.5*e*q))
            e = d;  d = p/q;
        else
            d = m;  e = m;
        end
    end % Interpolation
    
    % Next point
    a = b;
    fa = fb;
    if abs(d) > toler, b = b + d;
    elseif b > c, b = b - toler;
    else b = b + toler;
    end
    fb = itohabe_e(b,u_old,E,u,g,i,j,Dx,Dy,epsilon,aa,i1,i2,j1,j2,dt,energy);
end % Main loop

fval = fb; % b is the best value

end