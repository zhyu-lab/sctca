clc;
clear;
fclose all;

result_dir = './results/simu/';
method_names = {'scImpute' 'DeepImpute' 'bayNorm' 'GE-Impute' 'SCDD' 'scMultiGAN' 'TsImpute' 'MAGIC' 'CarDEC' 'scTCA'};

fontsize = 11;
markersize = 5;
linewidth = 1.5;

cmap = colorcube(length(method_names)+2);
cmap = cmap(2:end-1,:);
cmap = cmap([1:3 6 5 7:10 4],:);

legend_names = cell(1,2+length(method_names));
legend_names{1} = 'Original';
legend_names(2:length(method_names)+1) = method_names;
legend_names{end} = 'Ground truth';

ploidy = 2;
num_cell = 500;
num_clone = 7;
num_bin = 18000;
t = 1;
cols = num_clone+1;

ds = ['ploidy_' num2str(ploidy) '_tree_' num2str(t) '_clones_' num2str(num_clone) '_bins_' num2str(num_bin)];
disp(['ploting ' ds '_cells_' num2str(num_cell)])

cn_file = [base_dir '/dataset/simu/test_data/' ds '.cn'];
segs_all = read_real_segments(cn_file);

clone_cns = segs_all(:,4:end)';

assign_file = [base_dir '/dataset/simu/test_data/' ds '_cells_' num2str(num_cell) '.cell_assigns'];
cell_assigns = load(assign_file);
cell_cns = clone_cns(cell_assigns,:);
num_cell = size(cell_cns,1);

rc_file = [base_dir '/dataset/simu/test_data/' ds '_cells_' num2str(num_cell) '.rc'];
fid = fopen(rc_file,'r');
rc_ori = cell2mat(textscan(fid,repmat('%f',1,num_bin),'delimiter',','));
fclose(fid);

for c = 1:num_cell
    ploidy = mode(cell_cns(c,:));
    tv = cell_cns(c,:) == ploidy;
    rc = mean(rc_ori(c,tv));
    rc_ori(c,:) = rc_ori(c,:)/(2*rc/ploidy+eps);
end

figure(1);
clf;
tiledlayout(length(method_names)+2,cols,'TileSpacing','compact','Padding','compact')
sel_cells = zeros(1,cols);
for ii = 1:cols
    indxs = find(cell_assigns == ii);
    I = indxs(1);
    sel_cells(ii) = I;
end
max_rc = max(rc_ori(sel_cells,:),[],'all');
min_rc = min(rc_ori(sel_cells,:),[],'all');
max_cn = max(cell_cns(sel_cells,:),[],'all');
max_cn = floor(max_cn/2)*2+1;
for ii = 1:cols
    I = sel_cells(ii);
    y = rc_ori(I,:);
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

    y = cell_cns(I,:);
    nexttile(cols*(length(method_names)+1)+ii)
    set(gca,'FontSize',fontsize);
    plot(x,y,'-','linewidth',linewidth,'color',[0.5 0.5 0.5]);
    set(gca,'yticklabel',[]);
    axis([-inf inf 0.5 max_cn+0.5])
    set(gca,'xticklabel',[]);
end

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

    for c = 1:num_cell
        ploidy = mode(cell_cns(c,:));
        tv = cell_cns(c,:) == ploidy;
        rc = mean(rc_recons(c,tv));
        rc_recons(c,:) = rc_recons(c,:)/(2*rc/ploidy+eps);
    end

    max_rc = max(rc_recons(sel_cells,:),[],'all');
    min_rc = min(rc_recons(sel_cells,:),[],'all');
    for ii = 1:cols
        I = sel_cells(ii);
        y = rc_recons(I,:);
        x = 1:length(y);
        nexttile(m*cols+ii)
        set(gca,'FontSize',fontsize);
        plot(x,y,'.','markersize',markersize,'color',cmap(m,:));
        set(gca,'xticklabel',[],'yticklabel',[]);
        axis([-inf inf min_rc-max_rc/15 max_rc+max_rc/10])
    end
end

function results = read_real_segments(cn_file)

fid = fopen(cn_file,'r');
line = fgetl(fid);
fields = regexp(line,',','split');
num_seg = length(fields);
results = zeros(num_seg,3);
for i = 1:num_seg
    indx = find(fields{i} == ':');
    chr = str2double(fields{i}(1:indx-1));
    indx2 = find(fields{i} == '-');
    spos = str2double(fields{i}(indx+1:indx2-1));
    epos = str2double(fields{i}(indx2+1:end));
    results(i,:) = [chr spos epos];
end
while ~feof(fid)
    line = fgetl(fid);
    cn = str2double(regexp(line,',','split'));
    results = [results cn'];
end
fclose(fid);

end




