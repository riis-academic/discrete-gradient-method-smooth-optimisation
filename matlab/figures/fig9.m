clear 
close all

%MRI

% Load image, convert to grayscale and apply salt and pepper noise
image = 'MRI.png';
gl = imread(image);
gl = double(rgb2gray(gl));
gl = gl./max(max(gl));
rng(10);
g = imnoise(gl,'gaussian',0.2);
g = g(1:110,1:110);
K = ones(size(g));
[Ny,Nx] = size(g);
dx = 1/Nx;
dy = 1/Ny;
a = 0.2;
epsilon = 1E-8;
dt = 1./(a*2./realsqrt(epsilon)+1);
dt2 = [dt, 10*dt, 100*dt, 1000*dt];
dt1 = 1/10;
% dt1 = [1/10, 1, 10];
% dt = 1;
% T = 1000; % Max no. of time steps
T=50;
global count
count = 0;
[Ny,Nx] = size(g);

u = g;
s = 2;

xtol = 1E-14;
restol = 1E-18;

doplot = 1;

method = 'euler';

u2 = zeros([length(dt2), size(u),T+1]);
energy2 = zeros(length(dt2),T+1);

tic
[u1, energy1] = TV_MATLAB(g,K,u,a,s,epsilon,dt1,T,restol,xtol,doplot,'dg');
for i=1:length(dt2)
    [u2(i,:,:,:), energy2(i,:)] = TV_MATLAB(g,K,u,a,s,epsilon,dt2(i),T,restol,xtol,doplot,'euler');
end
toc

minE = min(min(energy1(:)),min(energy2(:)));

%%


figure
semilogy(1:6:40,(energy1(1:6:40)-minE)/(energy1(1,1)-minE),'ob',...
    'Markersize',15,'MarkerFaceColor','b','LineWidth',2.5)

hold on
semilogy(1:6:40,(energy2(1,1:6:40)-minE)/(energy2(1,1)-minE),'+r',...
    'Markersize',15,'MarkerFaceColor','r','LineWidth',2.5)
semilogy(1:6:40,(energy2(2,1:6:40)-minE)/(energy2(2,1)-minE),'*g',...
    'Markersize',15,'MarkerFaceColor','g','LineWidth',2.5)
semilogy(1:6:40,(energy2(3,1:6:40)-minE)/(energy2(3,1)-minE),'sm',...
    'Markersize',15,'MarkerFaceColor','m','LineWidth',2.5)
semilogy(1:6:40,(energy2(4,1:6:40)-minE)/(energy2(4,1)-minE),'dk',...
    'Markersize',15,'MarkerFaceColor','k','LineWidth',2.5)

semilogy(1:40,(energy1(1:40)-minE)/(energy1(1,1)-minE),'b','LineWidth',2.5)
semilogy(1:40,(energy2(1,1:40)-minE)/(energy2(1,1)-minE),'r','LineWidth',2.5)
semilogy(1:40,(energy2(2,1:40)-minE)/(energy2(2,1)-minE),'g','LineWidth',2.5)
semilogy(1:40,(energy2(3,1:40)-minE)/(energy2(3,1)-minE),'m','LineWidth',2.5)
semilogy(1:40,(energy2(4,1:40)-minE)/(energy2(4,1)-minE),'k','LineWidth',2.5)
hold off

lgd = legend('DG, \tau = 0.0250', 'CD, \tau = 0.0002', 'CD, \tau = 0.0025',...
    'CD, \tau = 0.0250', 'CD, \tau = 0.2500');
lgd.FontSize=20;
legend boxoff
ylabel('relative objective','fontsize',20)
xlabel('iterates','fontsize',20)
set(gca,'FontSize',20)