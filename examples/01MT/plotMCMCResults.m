% function plotMCMCResults
%
%--------------------------------------------------------------------------
surveytype = 'mt';
resultfile = ['posteriorModel-',surveytype,'.dat'];
fid = fopen(resultfile, 'r');
tmp = textscan(fid,'%f %f %f %f %f %f %f','HeaderLines',1);
fclose(fid);
results.mean   = 10 .^ tmp{2};
results.median = 10 .^ tmp{3};
results.mode   = 10 .^ tmp{4};
results.credmin = 10 .^ tmp{5};
results.credmax = 10 .^ tmp{6};
results.std     = 10 .^ tmp{7};
fclose(fid);

%%
m2km = 0.001;
depthBins = 10 .^ load(['depthBins-',surveytype,'.dat']);
rhoBins   = 10 .^ load(['rhoBins-',surveytype,'.dat']);

% histogram of number of layers
tmp = load(['nlayerHistogram-',surveytype,'.dat']);
nlayerHist = tmp(:,2);

% histogram of interface depths
depthHist = load(['depthHistogram-',surveytype,'.dat']);

% PPD of resistivity with depth
depthrhoHist = load(['depthrhoHistogram-',surveytype,'.dat']);

%% set plot range
depthplotlim = [0 2e4]*m2km;
rhoplotlim   = [0.1 1e4];
depthBins    = depthBins * m2km;

%% plot figures
figure(1);
fontsize = 11;
set(gcf,'Units','centimeters');
set(gcf,'Position',[10 2 30 14]);

%% PPD of the resistivity
subplot(1,4,1:2);
nsample = sum(depthrhoHist(:,1));
depthrhoHist = depthrhoHist / nsample;
pch = pcolor(rhoBins, depthBins, log10(depthrhoHist'));
set(pch, 'EdgeColor', 'none')
hold on
cm = colormap('jet');
cm(1:15,:) = [];

colormap(cm);
c = colorbar;
c.Location = 'westoutside';
c.Label.String = 'log_{10}Probability';
c.Label.FontSize = 10;
caxis([-3 0]);

linewidth = 1.2;
semilogx(results.mean,    depthBins, 'b-', 'LineWidth', linewidth);
hold on
semilogx(results.median,  depthBins, 'r-', 'LineWidth', linewidth);
semilogx(results.mode,    depthBins, 'k-', 'LineWidth', linewidth);
semilogx(results.credmin, depthBins, 'm-', 'LineWidth', linewidth);
semilogx(results.credmax, depthBins, 'm-', 'LineWidth', linewidth);

set(gca,'ydir','reverse','xscale','log','layer','top','box','on');
xlim(rhoplotlim);
ylim(depthplotlim);
xlabel('Resistivity [\Omegam]','Fontsize',fontsize);
ylabel('Depth [km]','Fontsize',fontsize);


%% histgram for the interface depths
subplot(1,4,3);
totalayer = sum(depthHist);
dtmp = depthHist / totalayer;
fill([0;dtmp;0],[0;depthBins;50],'b');
box on; grid on;
xlim( [0 0.1]);
ylim(depthplotlim);
set(gca,'ydir','reverse');
xlabel('Interface Probability','Fontsize',fontsize);
%
a = [0;dtmp;0];
b = [0;depthBins;50];
interfaceHist = [a b];

%% histogram for the number of layers
%
subplot(1,4,4);
bar(nlayerHist/nsample);
box on;
set(gca,'layer','top');
ylim([0 0.4]);
xlabel('Number of Layers','Fontsize',fontsize);
ylabel('Probability','Fontsize',fontsize);
