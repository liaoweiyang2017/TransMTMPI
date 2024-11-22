% function plotModel2dPPD
%
%% load data
modparam = load('modelParam_5layer.dat');

%% compute thickness of each layer
nsample = size(modparam, 1);
depth1d = 10.^modparam(:,6:end);
thick1d = [10.^modparam(:,6) diff(depth1d,1,2)];
thick1d = log10(thick1d);

%% plot 
hf = figure(1);
set(gcf,'Units','centimeters');
set(gcf,'Position',[8 2 24 16]);
nrow = 2;
ncol = 2;
mksize = 10;
linew  = 2;
capfs = 11;
axefs = 11;
%
k = 1;
subaxis(nrow,ncol,k,'ML',0.07,'MT',0.03,'MB',0.2,'MR',0.03);
rhotmp = modparam(:,k);
deptmp = modparam(:,k+5);
thktmp = thick1d(:,k);
h = histogram2(rhotmp,thktmp,'DisplayStyle','tile','ShowEmptyBins','off');
rhoRange = 2.0:0.02:3;
depRange = 2.4:0.02:3;
h.XBinEdges = rhoRange;
h.YBinEdges = depRange;
h.Normalization = 'probability';
set(h, 'EdgeColor', 'none')
ch = colorbar;
% ch.Label.String = 'Probability';
ch.Label.FontSize = 11;
ch.FontSize = 11;

caxis([0 0.05])
% xlabel('log_{10}Resistivity [\Omegam]','Fontsize',capfs);
ylabel('log_{10}Thickness [m]','Fontsize',capfs);
rho01 = log10(250);
dep01 = log10(600);
hold on
plot(rho01,dep01,'rx','MarkerSize',mksize,'LineWidth',linew);
set(gca,'layer','top','Fontsize',axefs);

text(2.03,2.96,'(a) first layer','Fontsize',13);

%
k = 2;
subaxis(nrow,ncol,k,'ML',0.06);
rhotmp = modparam(:,k);
deptmp = modparam(:,k+5);
thktmp = thick1d(:,k);
h = histogram2(rhotmp,thktmp,'DisplayStyle','tile','ShowEmptyBins','off');
rhoRange = 1.1:0.01:1.7;
depRange = 2.8:0.02:3.4;
h.XBinEdges = rhoRange;
h.YBinEdges = depRange;
h.Normalization = 'probability';
set(h, 'EdgeColor', 'none');
ch = colorbar;
% ch.Label.String = 'Probability';
ch.Label.FontSize = 11;
ch.FontSize = 11;

caxis([0 0.05])
% xlabel('log_{10}Resistivity [\Omegam]','Fontsize',capfs);
% ylabel('log_{10}Depth [m]','Fontsize',capfs);
rho02 = log10(25);
dep02 = log10(1400);
hold on
plot(rho02,dep02,'rx','MarkerSize',mksize,'LineWidth',linew);
set(gca,'layer','top','Fontsize',axefs);

text(1.12,3.36,'(b) second layer','Fontsize',13);

%
k = 3;
subaxis(nrow,ncol,k,'ML',0.07,'MT',0.02,'MB',0.1,'MR',0.03);
rhotmp = modparam(:,k);
deptmp = modparam(:,k+5);
thktmp = thick1d(:,k);
h = histogram2(rhotmp,thktmp,'DisplayStyle','tile','ShowEmptyBins','off');
rhoRange = 1.0:0.04:4.0;
depRange = 3.2:0.02:4.2;
h.XBinEdges = rhoRange;
h.YBinEdges = depRange;
h.Normalization = 'probability';
set(h, 'EdgeColor', 'none')
ch = colorbar;
% ch.Label.String = 'Probability';
ch.Label.FontSize = 11;
ch.FontSize = 11;

caxis([0 0.004])
xlabel('log_{10}Resistivity [\Omegam]','Fontsize',capfs);
ylabel('log_{10}Thickness [m]','Fontsize',capfs);
rho03 = log10(100);
dep03 = log10(4000);
hold on
plot(rho03,dep03,'rx','MarkerSize',mksize,'LineWidth',linew);
set(gca,'layer','top','Fontsize',axefs);

text(1.1,4.15,'(c) third layer','Fontsize',13);

%
k = 4;
subaxis(nrow,ncol,k,'ML',0.06);
rhotmp = modparam(:,k);
deptmp = modparam(:,k+5);
thktmp = thick1d(:,k);
h = histogram2(rhotmp,thktmp,'DisplayStyle','tile','ShowEmptyBins','off');
rhoRange = 0.2:0.02:1.4;
depRange = 3.0:0.02:4.4;
h.XBinEdges = rhoRange;
h.YBinEdges = depRange;
h.Normalization = 'probability';
set(h, 'EdgeColor', 'none')
ch = colorbar;
% ch.Label.String = 'Probability';
ch.Label.FontSize = 11;
ch.FontSize = 11;

caxis([0 0.004])
xlabel('log_{10}Resistivity [\Omegam]','Fontsize',capfs);
% ylabel('log_{10}Depth [m]','Fontsize',capfs);
rho04 = log10(10);
dep04 = log10(4000);
hold on
plot(rho04,dep04,'rx','MarkerSize',mksize,'LineWidth',linew);
set(gca,'layer','top','Fontsize',axefs);

text(0.23,4.32,'(d) fourth layer','Fontsize',13);

%% save figure
filename = 'layer2dPPD';
set(hf,'Units','Inches');
pos = get(hf,'Position');
set(hf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(hf,filename,'-dpdf','-r300')
print(hf,filename,'-dtiff','-r300')