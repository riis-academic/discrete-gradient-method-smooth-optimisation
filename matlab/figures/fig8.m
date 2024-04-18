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
epsilon = [1E-2,1E-4,1E-8];
dt2 = 1./(a*2./realsqrt(epsilon)+1);
dt1 = sqrt(dt2);
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

u1 = zeros([length(epsilon), size(u),T+1]);
energy1 = zeros(length(epsilon),T+1);
u2 = zeros([length(epsilon), size(u),T+1]);
energy2 = zeros(length(epsilon),T+1);

tic
for i=1:length(epsilon)
    [u1(i,:,:,:), energy1(i,:)] = TV_MATLAB(g,K,u,a,s,epsilon(i),dt1(i),T,restol,xtol,doplot,'dg');
    [u2(i,:,:,:), energy2(i,:)] = TV_MATLAB(g,K,u,a,s,epsilon(i),dt2(i),T,restol,xtol,doplot,'euler');
end
toc


%%

min1 = min(min(energy1(1,:)),min(energy2(1,:)));
min2 = 0.99999999999999*min(min(energy1(2,:)),min(energy2(2,:)));
min3 = 0.998*min(min(energy1(3,:)),min(energy2(3,:)));

figure
semilogy(1:4:size(energy1,2),(energy1(1,1:4:end)-min1)/(energy1(1,1)-min1),'ob',...
    'MarkerSize',15,'MarkerFaceColor','b','LineWidth',2.5)

hold on
semilogy(1:4:size(energy2,2),(energy2(1,1:4:end)-min1)/(energy2(1,1)-min1),'*r',...
    'MarkerSize',15,'MarkerFaceColor','r','LineWidth',2.5)
semilogy(1:size(energy1,2),(energy1(1,:)-min1)/(energy1(1,1)-min1),'b','LineWidth',2.5)
semilogy(1:size(energy2,2),(energy2(1,:)-min1)/(energy2(1,1)-min1),'r','LineWidth',2.5)
hold off

lgd = legend('DG, \epsilon = 10^{-2}', 'CD, \epsilon = 10^{-2}');
lgd.FontSize=20;
legend boxoff
ylabel('relative objective','fontsize',20)
xlabel('iterates','fontsize',20)
set(gca,'FontSize',20)
xlim([0,30])



figure
semilogy(1:15:size(energy1,2),(energy1(2,1:15:end)-min2)/(energy1(2,1)-min2),'ob',...
    'MarkerSize',15,'MarkerFaceColor','b','LineWidth',2.5)

hold on
semilogy(1:15:size(energy2,2),(energy2(2,1:15:end)-min2)/(energy2(2,1)-min2),'*r',...
    'MarkerSize',15,'MarkerFaceColor','r','LineWidth',2.5)
semilogy(1:size(energy1,2),(energy1(2,:)-min2)/(energy1(2,1)-min2),'b','LineWidth',2.5)
semilogy(1:size(energy2,2),(energy2(2,:)-min2)/(energy2(2,1)-min2),'r','LineWidth',2.5)
hold off

lgd = legend('DG, \epsilon = 10^{-4}', 'CD, \epsilon = 10^{-4}');
lgd.FontSize=20;
legend boxoff
ylabel('relative objective','fontsize',20)
xlabel('iterates','fontsize',20)
set(gca,'FontSize',20)
xlim([0,80])




figure
semilogy(1:15:size(energy1,2),(energy1(3,1:15:end)-min3)/(energy1(3,1)-min3),'ob',...
    'MarkerSize',15,'MarkerFaceColor','b','LineWidth',2.5)

hold on
semilogy(1:15:size(energy2,2),(energy2(3,1:15:end)-min3)/(energy2(3,1)-min3),'*r',...
    'MarkerSize',15,'MarkerFaceColor','r','LineWidth',2.5)
semilogy(1:size(energy1,2),(energy1(3,:)-min3)/(energy1(3,1)-min3),'b','LineWidth',2.5)
semilogy(1:size(energy2,2),(energy2(3,:)-min3)/(energy2(3,1)-min3),'r','LineWidth',2.5)
hold off

lgd = legend('DG, \epsilon = 10^{-8}', 'CD, \epsilon = 10^{-8}');
lgd.FontSize=20;
legend boxoff
ylabel('relative objective','fontsize',20)
xlabel('iterates','fontsize',20)
set(gca,'FontSize',20)
xlim([0,100])