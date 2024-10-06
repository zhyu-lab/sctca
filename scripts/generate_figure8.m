clc;
clear;
fclose all;

result_dir = './results/real/';
method_names = {'DeepImpute' 'bayNorm' 'GE-Impute' 'SCDD' 'MAGIC' 'CarDEC' 'scTCA'};
datasets = 'ABCDE';

fontsize = 15;
markersize = 50;

cmap_all = colormap(turbo);

rng(8);

fi = 1;

for i = 1:length(datasets)
    ds = datasets(i);
    disp(['ploting ' ds])

    bin_file = [result_dir ds '.20k.filtered.bins.txt'];
    fid = fopen(bin_file,'r');
    fgetl(fid);
    line = fgetl(fid);
    fields = regexp(line,',','split');
    num_bin = length(fields);
    fclose(fid);
    
    format = [repmat('%f%*f%*f',1,floor(num_bin/3)) repmat('%*f',1,rem(num_bin,3))];

    assign_file = [result_dir ds '.cell_assigns'];
    cell_assigns = load(assign_file);
    clone_uids = unique(cell_assigns);
    num_clone = length(clone_uids);
    
    cmap = zeros(num_clone,3);
    legend_names = cell(1,num_clone);
    m = floor(size(cmap_all,1)/(num_clone-1))-1;
    for n = 1:num_clone
        k = 1+(n-1)*m;
        cmap(n,:) = cmap_all(k,:);
        legend_names{n} = ['Subpopulation ' num2str(n)];
    end

    rc_file = [result_dir ds '.20k.filtered.txt'];
    fid = fopen(rc_file,'r');
    rc_ori = cell2mat(textscan(fid,format,'delimiter',','));
    fclose(fid);
    
    colors = [];
    indxs_sorted = [];
    label_n = [];
    for n = 1:num_clone
        id = clone_uids(n);
        indxs = find(id == cell_assigns);
        indxs_sorted = [indxs_sorted indxs];
        label_n = [label_n ones(1,length(indxs))*n];
        colors = [colors; repmat(cmap(n,:),length(indxs),1)];
    end

    options_tsne = struct('NumDimensions',2,'NumPCAComponents',50);
    data_tsne = tsne(rc_ori(indxs_sorted,:),'Options',options_tsne);
    clear rc_ori;
    
    figure(fi);
    fi = fi+1;
    clf;
    subplot(2,4,1);
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
    for n = 1:num_clone
        h = [h scatter(nan,nan,30,cmap(n,:),'filled')];
    end
    legend(h,legend_names,'FontSize',fontsize-2)

    for m = 1:length(method_names)
        rc_file = [result_dir method_names{m} '/' ds '/result.txt'];
        if strcmp(method_names{m},'scTCA')==1
            rc_file = [result_dir method_names{m} '/' ds '/corrected_rc.txt'];
        end
        if strcmp(method_names{m},'SCDD')==1
            rc_file = [result_dir method_names{m} '/' ds '/result_3.txt'];
        end
        
        fid = fopen(rc_file,'r');
        if strcmp(method_names{m},'TsImpute')==1
            rc_recons = cell2mat(textscan(fid,format,'delimiter',','));
        else
            rc_recons = cell2mat(textscan(fid,format));
        end
        fclose(fid);

        data_tsne = tsne(rc_recons(indxs_sorted,:),'Options',options_tsne);
        clear rc_recons;

        subplot(2,4,m+1);
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
end






