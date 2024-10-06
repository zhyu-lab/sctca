clc;
clear;
fclose all;

datasets = 'ABCDE';

load results_real.mat;

method_names = {'DeepImpute' 'GE-Impute' 'SCDD' 'MAGIC' 'CarDEC' 'bayNorm' 'scTCA'};

fontsize = 19;
linewidth = 1.5;
cmap = colorcube(10+2);
cmap = cmap(2:end-1,:);
cmap = cmap([1:3 6 5 7:10 4],:);
cmap = cmap([2:5 8:10],:);

m_indxs = [1 6 2:5 7];

figure(1);
clf;
subplot(2,2,1)
y = rho_all;
y = y(:,m_indxs);
x = 1:length(datasets);
hold on
set(gca,'box','on','xgrid','on','fontsize',fontsize);
h = bar(x,y,1);
for k = 1:length(method_names)
    h(k).FaceColor = cmap(k,:);
end
set(gca,'xtick',x,'xticklabel',[])
ylabel('Spearman Coefficient','fontsize',fontsize+2);
axis([0.5 max(x)+0.5 0 1])
set(gca,'ytick',0:0.2:1.0,'ygrid','on');

for k = 2:max(x)
    x = (k+k-1)/2;
    plot([x x],[0 1],'-','color',[0.6 0.6 0.6],'linewidth',0.5);
end
legend(method_names(m_indxs),'fontsize',fontsize);

y = delta_rho_all;
y = y(:,m_indxs);
x = 1:length(datasets);
subplot(2,2,2)
hold on
set(gca,'box','on','xgrid','on','fontsize',fontsize);
h = bar(x,y,1);
for k = 1:length(method_names)
    h(k).FaceColor = cmap(k,:);
end
set(gca,'xtick',x,'xticklabel',[])
ylabel('\it\Delta \rmSpearman','fontsize',fontsize+2);
axis([0.5 max(x)+0.5 0 1])
set(gca,'ytick',0:0.2:1,'ygrid','on');

for k = 2:max(x)
    x = (k+k-1)/2;
    plot([x x],[-0.25 1],'-','color',[0.6 0.6 0.6],'linewidth',0.5);
end

y = delta_silhouette_all;
y = y(:,m_indxs);
x = 1:length(datasets);
subplot(2,2,3)
hold on
set(gca,'box','on','xgrid','on','fontsize',fontsize);
h = bar(x,y,1);
for k = 1:length(method_names)
    h(k).FaceColor = cmap(k,:);
end
set(gca,'xtick',x,'xticklabel',mat2cell(datasets,1,ones(1,length(datasets))))
xlabel('Dataset','fontsize',fontsize+2);
ylabel('\it\Delta \rmSilhouette','fontsize',fontsize+2);
axis([0.5 max(x)+0.5 0 1])
set(gca,'ytick',0:0.2:1.0,'ygrid','on');

for k = 2:max(x)
    x = (k+k-1)/2;
    plot([x x],[0 1],'-','color',[0.6 0.6 0.6],'linewidth',0.5);
end

y = delta_cv_all;
y = y(:,m_indxs);
x = 1:length(datasets);
subplot(2,2,4)
hold on
set(gca,'box','on','xgrid','on','fontsize',fontsize);
h = bar(x,y,1);
for k = 1:length(method_names)
    h(k).FaceColor = cmap(k,:);
end
set(gca,'xtick',x,'xticklabel',mat2cell(datasets,1,ones(1,length(datasets))))
xlabel('Dataset','fontsize',fontsize+2);
ylabel('\it\Delta \rmCV','fontsize',fontsize+2);
axis([0.5 max(x)+0.5 0 1])
set(gca,'ytick',0:0.2:1.0,'ygrid','on');

for k = 2:max(x)
    x = (k+k-1)/2;
    plot([x x],[0 1],'-','color',[0.6 0.6 0.6],'linewidth',0.5);
end





