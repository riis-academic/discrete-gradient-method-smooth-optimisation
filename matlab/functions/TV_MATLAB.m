function [u_collection, energy, fevals] = TV_MATLAB(g,K,u,a,s,epsilon,dt,T,restol,xtol,doplot,method)

global count

[Ny,Nx] = size(g);
if doplot
    figure(1);
    imagesc(g);
    colormap(gray);
end

energy = zeros(1,T+1);
energy(1) = energyfxn(u,g,a,epsilon);
fevals = zeros(1,T+1);
fevals(1) = 0;

u_collection = zeros(Ny,Nx,T+1);
u_collection(:,:,1) = u;
residual = 1;
t1 = tic;
iters = 1;
for tstep = 1:T
    if false %residual < restol
        break
    end
    if strcmp(method,'dg')
        disp([num2str(round(tstep/T*100)) '%'])
        u = dgstep_e(u,K,g,dt,a,s,epsilon,xtol,energy(tstep));
    elseif strcmp(method,'euler')
        u = eulerstep(u,g,dt,a,epsilon);
    elseif strcmp(method,'eulerls')
        u = eulerlsstep(u,g,dt,a,epsilon,energy(tstep));
    else
        error('Method not recognised.')
    end
    u = u(2:end-1,2:end-1);
    u_collection(:,:,tstep+1) = u;
    energy(tstep+1) = energyfxn(u,g,a,epsilon);
    fevals(tstep+1) = count;
    residual = (energy(tstep) - energy(tstep+1))/energy(1);
    if doplot && mod(tstep,10)==0
        figure(5), imagesc(u); colormap(gray);
        pause(0.01)
        figure(4), semilogy(energy(1:tstep+1)/energy(1));
        pause(0.01)
    end
    iters = iters+1;
end
t2 = toc(t1);
disp(['Time used: ' num2str(t2)])
disp(['Stopping energy: ' num2str(energy(iters))])
disp(['Iterations: ' num2str(iters)])
end