clc;
clear;
fclose all;

result_dir = './results/real/';
method_names = {'DeepImpute' 'bayNorm' 'GE-Impute' 'SCDD' 'MAGIC' 'CarDEC' 'scTCA'};
datasets = 'ABCDE';

fontsize = 11;
markersize = 5;
linewidth = 1.5;

cmap = colorcube(10+2);
cmap = cmap(2:end-1,:);
cmap = cmap([1:3 6 5 7:10 4],:);
cmap = cmap([2:5 8:10],:);

fi = 1;

legend_names = cell(1,2+length(method_names));
legend_names{1} = 'Original';
legend_names(2:length(method_names)+1) = method_names;
legend_names{end} = 'Ground truth';

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
    
    cn_file = [result_dir ds '.20k.cn.txt'];
    fid = fopen(cn_file,'r');
    cell_cns = cell2mat(textscan(fid,format,'delimiter',','));
    fclose(fid);
    num_cell = size(cell_cns,1);

    assign_file = [result_dir ds '.cell_assigns'];
    cell_assigns = load(assign_file);
    clone_uids = unique(cell_assigns);
    num_clone = length(clone_uids);
    cols = num_clone;
    
    sel_cells = zeros(1,cols);
    for ii = 1:cols
        cell_indxs = find(cell_assigns == clone_uids(ii));
        m_cns = mean(cell_cns(cell_indxs,:));
        dists = sum((cell_cns(cell_indxs,:)-repmat(m_cns,length(cell_indxs),1)).^2,2);
        [~, j] = min(dists);
        sel_cells(ii) = cell_indxs(j);
    end

    rc_file = [result_dir ds '.20k.filtered.txt'];
    fid = fopen(rc_file,'r');
    rc_ori = cell2mat(textscan(fid,format,'delimiter',','));
    fclose(fid);
    
    for c = 1:num_cell
        ploidy = mode(cell_cns(c,:));
        tv = cell_cns(c,:) == ploidy;
        rc = mean(rc_ori(c,tv));
        rc_ori(c,:) = rc_ori(c,:)/(2*rc/ploidy+eps);
    end
    
    rc_ori = rc_ori(sel_cells,:);
    cn_real = cell_cns(sel_cells,:);
    
    max_rc = max(rc_ori,[],'all');
    min_rc = min(rc_ori,[],'all');
    max_cn = max(cn_real,[],'all');
    max_cn = floor(max_cn/2)*2+1;
    
    figure(fi);
    fi = fi+1;
    clf;
    tiledlayout(length(method_names)+2,cols,'TileSpacing','compact','Padding','compact')
    for ii = 1:cols
        y = rc_ori(ii,:);
        x = 1:length(y);
        nexttile(ii)
        set(gca,'FontSize',fontsize);
        plot(x,y,'.','markersize',markersize,'color',[1.0 0.8 0.5]);
        if ii == 1
            hold on
            h = [];
            h(1) = scatter(nan,nan,30,[1.0 0.8 0.5],'filled');
            for jj = 1:length(method_names)
                h(jj+1) = scatter(nan,nan,30,cmap(jj,:),'filled');
            end
            h(length(method_names)+2) = scatter(nan,nan,30,[0.5 0.5 0.5],'filled');
            legend(h,legend_names,'FontSize',fontsize+8)
        end

        set(gca,'xticklabel',[],'yticklabel',[]);
        axis([-inf inf min_rc-max_rc/15 max_rc+max_rc/10])
        title(['Cluster ' num2str(ii)],'FontSize',fontsize+3);


        y = cn_real(ii,:);
        nexttile(cols*(length(method_names)+1)+ii)
        set(gca,'FontSize',fontsize);
        plot(x,y,'-','linewidth',linewidth,'color',[0.5 0.5 0.5]);
        set(gca,'yticklabel',[]);
        axis([-inf inf 0.5 max_cn+0.5])
        set(gca,'xticklabel',[]);
    end

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
        
        for c = 1:num_cell
            ploidy = mode(cell_cns(c,:));
            tv = cell_cns(c,:) == ploidy;
            rc = mean(rc_recons(c,tv));
            rc_recons(c,:) = rc_recons(c,:)/(2*rc/ploidy+eps);
        end
        
        rc_recons = rc_recons(sel_cells,:);

        max_rc = max(rc_recons,[],'all');
        min_rc = min(rc_recons,[],'all');
        for ii = 1:cols
            y = rc_recons(ii,:);
            x = 1:length(y);
            nexttile(m*cols+ii)
            set(gca,'FontSize',fontsize);
            plot(x,y,'.','markersize',markersize,'color',cmap(m,:));
            set(gca,'xticklabel',[],'yticklabel',[]);
            axis([-inf inf min_rc-max_rc/15 max_rc+max_rc/10])
        end
    end
end









