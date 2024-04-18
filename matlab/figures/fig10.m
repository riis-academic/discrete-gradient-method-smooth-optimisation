clear 
close all

global count energyVec
count = 0;
energyVec = [];

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
dt2 = 1;
dt1 = 1/10;
T=100;
[Ny,Nx] = size(g);

u = g;
s = 2;

xtol = 1E-14;
restol = 1E-18;

doplot = 1;

method = 'eulerls';

tic
[u1, energy1, fevals1] = TV_MATLAB(g,K,u,a,s,epsilon,dt1,T,restol,xtol,doplot,'dg');


[u2, energy2, fevals2] = TV_MATLAB(g,K,u,a,s,epsilon,dt2,T,restol,xtol,doplot,'eulerls');

toc

%%
minE = 0.99*min(min(energy1(:)),min(energy2(:)));
cumFevals1 = cumsum(fevals1);
cumFevals2 = cumsum(fevals2);


energyAlt1 = (energy1-minE)/(energy1(1)-minE);
energyAlt2 = (energy2-minE)/(energy2(1)-minE);

%%

N1 = round(length(energyAlt1)/8);
N2 = round(length(energyAlt2)/8);


figure
semilogy(1:N1:length(energyAlt1),energyAlt1(1:N1:end),'ob','MarkerSize',15,...
    'MarkerFaceColor','b','LineWidth',2.5)

hold on
semilogy(1:N2:length(energyAlt2),energyAlt2(1:N2:end),'+r','MarkerSize',15,...
    'MarkerFaceColor','r','LineWidth',2.5)


semilogy(1:length(energyAlt1),energyAlt1,'b','LineWidth',2.5)
semilogy(1:length(energyAlt2),energyAlt2,'r','LineWidth',2.5)
hold off

lgd = legend('DG','LS');
lgd.FontSize=18;
ylabel('relative objective','fontsize',18)
xlabel('iterates','fontsize',18)
set(gca,'FontSize',18)