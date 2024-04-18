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
% g = g(1:110,1:110);
K = ones(size(g));
[Ny,Nx] = size(g);
dx = 1/Nx;
dy = 1/Ny;
a = 0.2;
epsilon = 1E-8;
dt2 = 1./(a*2./realsqrt(epsilon)+1);
dt1 = [1/4000,1/50,1/10,1];
% dt1 = [1/10, 1, 10];
% dt = 1;
% T = 1000; % Max no. of time steps
T=100;
global count
count = 0;
[Ny,Nx] = size(g);

u = g;
s = 2;

xtol = 1E-14;
restol = 1E-18;

doplot = 1;

method = 'euler';

u1 = zeros([length(dt1), size(u),T+1]);
energy1 = zeros(length(dt1),T+1);

tic
[u2, energy2] = TV_MATLAB(g,K,u,a,s,epsilon,dt2,T,restol,xtol,doplot,'euler');
for i=1:length(dt1)
    [u1(i,:,:,:), energy1(i,:)] = TV_MATLAB(g,K,u,a,s,epsilon,dt1(i),T,restol,xtol,doplot,'dg');
end
toc

%%

minE = 0.9993*min(min(energy1(:)),min(energy2(:)));

fig = figure;
semilogy(1:15:size(energy1,2),(energy1(1,1:15:end)-minE)/(energy1(1,1)-minE),...
    'ob','MarkerSize',15,'MarkerFaceColor','b','LineWidth',2.5)
hold on
semilogy(1:15:size(energy1,2),(energy1(2,1:15:end)-minE)/(energy1(2,1)-minE),...
    '+g','MarkerSize',15,'MarkerFaceColor','g','LineWidth',2.5)
semilogy(1:15:size(energy1,2),(energy1(3,1:15:end)-minE)/(energy1(3,1)-minE),...
    '*k','MarkerSize',15,'MarkerFaceColor','k','LineWidth',2.5)
semilogy(1:15:size(energy1,2),(energy1(4,1:15:end)-minE)/(energy1(4,1)-minE),...
    'dm','MarkerSize',15,'MarkerFaceColor','m','LineWidth',2.5)
semilogy(7:15:size(energy2,2),(energy2(7:15:end)-minE)/(energy2(1)-minE),'sr',...
    'MarkerSize',15,'MarkerFaceColor','r','LineWidth',2.5)

semilogy(1:size(energy1,2),(energy1(1,:)-minE)/(energy1(1,1)-minE),'b','LineWidth',2.5)
semilogy(1:size(energy2,2),(energy2-minE)/(energy2(1)-minE),'--r','LineWidth',2.5)
semilogy(1:size(energy1,2),(energy1(2,:)-minE)/(energy1(2,1)-minE),'g','LineWidth',2.5)
semilogy(1:size(energy1,2),(energy1(3,:)-minE)/(energy1(3,1)-minE),'k','LineWidth',2.5)
semilogy(1:size(energy1,2),(energy1(4,:)-minE)/(energy1(4,1)-minE),'m','LineWidth',2.5)
hold off

lgd = legend('DG, \tau = 1/40', 'DG, \tau = 1/10', 'DG, \tau = 1', 'DG, \tau = 2', 'CD');
legend boxoff
lgd.FontSize=20;
ylabel('relative objective','fontsize',20)
xlabel('iterates','fontsize',20)
set(gca,'FontSize',20)