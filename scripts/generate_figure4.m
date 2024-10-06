clc;
clear;
fclose all;

result_dir = './results/simu/';
method_names = {'scImpute' 'DeepImpute' 'bayNorm' 'GE-Impute' 'SCDD' 'scMultiGAN' 'TsImpute' 'MAGIC' 'CarDEC' 'scTCA'};

fontsize = 15;
markersize = 50;

cmap_all = colormap(turbo);

rng(8);

fi = 1;

ploidy = 2;
num_cell = 1000;
num_clone = 11;
num_bin = 18000;
t = 2;

cmap = zeros(num_clone+1,3);
legend_names = cell(1,num_clone+1);
m = floor(size(cmap_all,1)/num_clone)-1;
for n = 1:num_clone+1
    k = 1+(n-1)*m;
    cmap(n,:) = cmap_all(k,:);
    legend_names{n} = ['Subpopulation ' num2str(n)];
end
ds = ['ploidy_' num2str(ploidy) '_tree_' num2str(t) '_clones_' num2str(num_clone) '_bins_' num2str(num_bin)];
disp(['ploting ' ds '_cells_' num2str(num_cell)])

assign_file = ['data/' ds '_cells_' num2str(num_cell) '.cell_assigns'];
cell_assigns = load(assign_file);

colors = [];
indxs_sorted = [];
label_n = [];
for n = 1:num_clone+1
    indxs = find(n == cell_assigns);
    indxs_sorted = [indxs_sorted indxs];
    label_n = [label_n ones(1,length(indxs))*n];
    colors = [colors; repmat(cmap(n,:),length(indxs),1)];
end

rc_file = ['data/' ds '_cells_' num2str(num_cell) '.rc'];
fid = fopen(rc_file,'r');
rc_ori = cell2mat(textscan(fid,repmat('%f',1,num_bin),'delimiter',','));
fclose(fid);

options_tsne = struct('NumDimensions',2,'NumPCAComponents',50);
data_tsne = tsne(rc_ori(indxs_sorted,:),'Options',options_tsne);
clear rc_ori;

figure(1);
clf;
subplot(3,4,1)
hold on
set(gca,'FontSize',fontsize);
scatter(data_tsne(:,1),data_tsne(:,2),markersize,colors,'filled');
set(gca,'box','on');
xlabel('tSNE1','FontSize',fontsize+1);
ylabel('tSNE2','FontSize',fontsize+1);
set(gca,'xticklabel',[],'yticklabel',[]);
tmp1 = (max(data_tsne(:,1))-min(data_tsne(:,1)))/10;
tmp2 = (max(data_tsne(:,2))-min(data_tsne(:,2)))/10;
axis([min(data_tsne(:,1))-tmp1 max(data_tsne(:,1))+tmp1 min(data_tsne(:,2))-tmp2 max(data_tsne(:,2))+tmp2])
title('Original','FontSize',fontsize+3);
ax = gca;
ax.TitleHorizontalAlignment = 'left';

h = [];
for n = 1:num_clone+1
    h = [h scatter(nan,nan,30,cmap(n,:),'filled')];
end
legend(h,legend_names,'FontSize',fontsize-2)

for m = 1:length(method_names)
    rc_file = [result_dir method_names{m} '/' ds '_cells_' num2str(num_cell) '/result.txt'];
    if strcmp(method_names{m},'scTCA')==1
        rc_file = [result_dir method_names{m} '/' ds '_cells_' num2str(num_cell) '/corrected_rc.txt'];
    end
    if strcmp(method_names{m},'SCDD')==1
        rc_file = [result_dir method_names{m} '/' ds '_cells_' num2str(num_cell) '/result_3.txt'];
    end
    fid = fopen(rc_file,'r');
    if strcmp(method_names{m},'TsImpute')==1
        rc_recons = cell2mat(textscan(fid,repmat('%f',1,num_bin),'delimiter',','));
    else
        rc_recons = cell2mat(textscan(fid,repmat('%f',1,num_bin)));
    end
    fclose(fid);

    data_tsne = tsne(rc_recons(indxs_sorted,:),'Options',options_tsne);
    clear rc_recons;

    subplot(3,4,m+1)
    hold on
    set(gca,'FontSize',fontsize);
    scatter(data_tsne(:,1),data_tsne(:,2),markersize,colors,'filled');
    set(gca,'box','on');
    xlabel('tSNE1','FontSize',fontsize+1);
    ylabel('tSNE2','FontSize',fontsize+1);
    set(gca,'xticklabel',[],'yticklabel',[]);
    tmp1 = (max(data_tsne(:,1))-min(data_tsne(:,1)))/10;
    tmp2 = (max(data_tsne(:,2))-min(data_tsne(:,2)))/10;
    axis([min(data_tsne(:,1))-tmp1 max(data_tsne(:,1))+tmp1 min(data_tsne(:,2))-tmp2 max(data_tsne(:,2))+tmp2])
    title(method_names{m},'FontSize',fontsize+3);
    ax = gca;
    ax.TitleHorizontalAlignment = 'left';
end





