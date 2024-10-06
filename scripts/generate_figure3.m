clc;
clear;
fclose all;

num_cells = [500 1000];
num_clones = [7 9 11];
num_bins = [12000 18000 24000];

load results_simu.mat;

pth = './boxplot2/';
addpath(fullfile(pth, 'boxplot2'));

method_names = {'scImpute' 'DeepImpute' 'GE-Impute' 'SCDD' 'scMultiGAN' 'TsImpute' 'MAGIC' 'bayNorm' 'CarDEC' 'scTCA'};

fontsize = 19;
linewidth = 1.5;
cmap = colorcube(length(method_names)+2);
cmap = cmap(2:end-1,:);
cmap = cmap([1:3 6 5 7:10 4],:);

figure(1);
clf;

m_indxs = [1 2 8 3:7 9 10];

for i = 1:length(num_cells)
    num_cell = num_cells(i);
    for j = 1:length(num_clones)
        num_clone = num_clones(j);
        y = rho_all{i,j};
        y = y(:,m_indxs,:);
        x = 1:length(num_bins);
        subplot(length(num_clones),length(num_cells),(j-1)*length(num_cells)+i)
        hold on
        set(gca,'box','on','xgrid','on','fontsize',fontsize);
        h = boxplot2(y,x,'barwidth',0.98);
        for ii = 1:length(method_names)
            structfun(@(x) set(x(ii,:),'color',cmap(ii,:), ...
                'markeredgecolor',cmap(ii,:),'linewidth',linewidth),h);
        end
        set([h.lwhis h.uwhis],'linestyle','-');
        set(h.out,'marker','.');
        set(gca,'xtick',x)
        if j == length(num_clones)
            xlabel('Number of bins','fontsize',fontsize+2);
        end
        ylabel('Spearman Coefficient','fontsize',fontsize+2);
        title([num2str(num_cell) ' cells, ' num2str(num_clone) ' clones'],'fontsize',fontsize+2);
        axis([0.5 length(num_bins)+0.5 -0.25 1])
        set(gca,'ytick',-0.25:0.25:1.0,'ygrid','on');
        if j == length(num_clones)
            set(gca,'xticklabel',num_bins);
        else
            set(gca,'xticklabel',[]);
        end

        for k = 2:length(num_bins)
            x = (k+k-1)/2;
            plot([x x],[-0.25 1],'-','color',[0.6 0.6 0.6],'linewidth',0.5);
        end
        if i == length(num_cells) && j == length(num_clones)
            legend(method_names(m_indxs),'fontsize',fontsize);
        end
    end
end

figure(2);
clf;
for i = 1:length(num_cells)
    num_cell = num_cells(i);
    for j = 1:length(num_clones)
        num_clone = num_clones(j);
        y = delta_rho_all{i,j};
        y = y(:,m_indxs,:);
        x = 1:length(num_bins);
        subplot(length(num_clones),length(num_cells),(j-1)*length(num_cells)+i)
        hold on
        set(gca,'box','on','xgrid','on','fontsize',fontsize);
        h = boxplot2(y,x,'barwidth',0.98);
        for ii = 1:length(method_names)
            structfun(@(x) set(x(ii,:),'color',cmap(ii,:), ...
                'markeredgecolor',cmap(ii,:),'linewidth',linewidth),h);
        end
        set([h.lwhis h.uwhis],'linestyle','-');
        set(h.out,'marker','.');
        set(gca,'xtick',x)
        if j == length(num_clones)
            xlabel('Number of bins','fontsize',fontsize+2);
        end
        ylabel('\it\Delta \rmSpearman','fontsize',fontsize+2);
        title([num2str(num_cell) ' cells, ' num2str(num_clone) ' clones'],'fontsize',fontsize+2);
        axis([0.5 length(num_bins)+0.5 0 0.6])
        set(gca,'ytick',0:0.2:0.6,'ygrid','on');
        if j == length(num_clones)
            set(gca,'xticklabel',num_bins);
        else
            set(gca,'xticklabel',[]);
        end

        for k = 2:length(num_bins)
            x = (k+k-1)/2;
            plot([x x],[0 0.6],'-','color',[0.6 0.6 0.6],'linewidth',0.5);
        end
        if i == length(num_cells) && j == length(num_clones)
            legend(method_names(m_indxs),'fontsize',fontsize);
        end
    end
end

figure(3);
clf;
for i = 1:length(num_cells)
    num_cell = num_cells(i);
    for j = 1:length(num_clones)
        num_clone = num_clones(j);
        y = delta_silhouette_all{i,j};
        y = y(:,m_indxs,:);
        x = 1:length(num_bins);
        subplot(length(num_clones),length(num_cells),(j-1)*length(num_cells)+i)
        hold on
        set(gca,'box','on','xgrid','on','fontsize',fontsize);
        h = boxplot2(y,x,'barwidth',0.98);
        for ii = 1:length(method_names)
            structfun(@(x) set(x(ii,:),'color',cmap(ii,:), ...
                'markeredgecolor',cmap(ii,:),'linewidth',linewidth),h);
        end
        set([h.lwhis h.uwhis],'linestyle','-');
        set(h.out,'marker','.');
        set(gca,'xtick',x)
        if j == length(num_clones)
            xlabel('Number of bins','fontsize',fontsize+2);
        end
        ylabel('\it\Delta \rmSilhouette','fontsize',fontsize+2);
        title([num2str(num_cell) ' cells, ' num2str(num_clone) ' clones'],'fontsize',fontsize+2);
        axis([0.5 length(num_bins)+0.5 -0.25 1])
        set(gca,'ytick',-0.25:0.25:1.0,'ygrid','on');
        if j == length(num_clones)
            set(gca,'xticklabel',num_bins);
        else
            set(gca,'xticklabel',[]);
        end

        for k = 2:length(num_bins)
            x = (k+k-1)/2;
            plot([x x],[-0.25 1],'-','color',[0.6 0.6 0.6],'linewidth',0.5);
        end
        if i == length(num_cells) && j == length(num_clones)
            legend(method_names(m_indxs),'fontsize',fontsize);
        end
    end
end

figure(4);
clf;
for i = 1:length(num_cells)
    num_cell = num_cells(i);
    for j = 1:length(num_clones)
        num_clone = num_clones(j);
        y = delta_cv_all{i,j};
        y = y(:,m_indxs,:);
        x = 1:length(num_bins);
        subplot(length(num_clones),length(num_cells),(j-1)*length(num_cells)+i)
        hold on
        set(gca,'box','on','xgrid','on','fontsize',fontsize);
        h = boxplot2(y,x,'barwidth',0.98);
        for ii = 1:length(method_names)
            structfun(@(x) set(x(ii,:),'color',cmap(ii,:), ...
                'markeredgecolor',cmap(ii,:),'linewidth',linewidth),h);
        end
        set([h.lwhis h.uwhis],'linestyle','-');
        set(h.out,'marker','.');
        set(gca,'xtick',x)
        if j == length(num_clones)
            xlabel('Number of bins','fontsize',fontsize+2);
        end
        ylabel('\it\Delta \rmCV','fontsize',fontsize+2);
        title([num2str(num_cell) ' cells, ' num2str(num_clone) ' clones'],'fontsize',fontsize+2);
        axis([0.5 length(num_bins)+0.5 0 1])
        set(gca,'ytick',0:0.25:1.0,'ygrid','on');
        if j == length(num_clones)
            set(gca,'xticklabel',num_bins);
        else
            set(gca,'xticklabel',[]);
        end

        for k = 2:length(num_bins)
            x = (k+k-1)/2;
            plot([x x],[-0.25 1],'-','color',[0.6 0.6 0.6],'linewidth',0.5);
        end
        if i == length(num_cells) && j == length(num_clones)
            legend(method_names(m_indxs),'fontsize',fontsize);
        end
    end
end





