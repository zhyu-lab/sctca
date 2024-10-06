clc;
clear;
fclose all;

base_dir = './denoising_rc';
method_names = {'DeepImpute' 'GE-Impute' 'SCDD' 'MAGIC' 'CarDEC' 'bayNorm' 'scTCA'};
datasets = 'ABCDE';

rho_all = zeros(length(datasets),length(method_names));
delta_rho_all = zeros(length(datasets),length(method_names));
delta_silhouette_all = zeros(length(datasets),length(method_names));
delta_cv_all = zeros(length(datasets),length(method_names));

for i = 1:length(datasets)
    ds = datasets(i);
    disp(['processing ' ds])

    bin_file = [base_dir '/dataset/real/chisel/' ds '.20k.filtered.bins.txt'];
    fid = fopen(bin_file,'r');
    fgetl(fid);
    line = fgetl(fid);
    fields = regexp(line,',','split');
    num_bin = length(fields);
    fclose(fid);

    cn_file = [base_dir '/dataset/real/chisel/' ds '.20k.cn.txt'];
    fid = fopen(cn_file,'r');
    cell_cns = cell2mat(textscan(fid,repmat('%f',1,num_bin),'delimiter',','));
    num_cell = size(cell_cns,1);
    fclose(fid);

    assign_file = [base_dir '/dataset/real/chisel/' ds '.cell_assigns'];
    cell_assigns = load(assign_file);

    rc_file = [base_dir '/dataset/real/chisel/' ds '.20k.filtered.txt'];
    fid = fopen(rc_file,'r');
    rc_ori = cell2mat(textscan(fid,repmat('%f',1,num_bin),'delimiter',','));
    fclose(fid);
    
    for c = 1:num_cell
        ploidy = mode(cell_cns(c,:));
        tv = cell_cns(c,:) == ploidy;
        rc = mean(rc_ori(c,tv));
        rc_ori(c,:) = rc_ori(c,:)/(2*rc/ploidy+eps);
    end

    dropout_tv = rc_ori == 0;

    tmp = corr([cell_cns(:) rc_ori(:)],'type','Spearman');
    rho2 = tmp(1,2);
    sh2 = mean(silhouette(rc_ori,cell_assigns,'cosine'));
    
    % coefficient of variation
    cvs_all_ori = [];
    for c = 1:num_cell
        u_cns = unique(cell_cns(c,:));
        cvs = zeros(1,length(u_cns));
        for n = 1:length(u_cns)
            tv = cell_cns(c,:) == u_cns(n);
            cvs(n) = std(rc_ori(c,tv))/mean(rc_ori(c,tv));
        end
        cvs_all_ori = [cvs_all_ori cvs];
    end
    
    clear rc_ori;
    
    for m = 1:length(method_names)
        rc_file = [base_dir '/results/real/chisel/' method_names{m} '/' ds '/result.txt'];
        if strcmp(method_names{m},'scTCA')==1
            rc_file = [base_dir '/results/real/chisel/' method_names{m} '/' ds '/corrected_rc.txt'];
        end
        if strcmp(method_names{m},'SCDD')==1
            rc_file = [base_dir '/results/real/chisel/' method_names{m} '/' ds '/result_3.txt'];
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

        tmp = corr([cell_cns(dropout_tv) rc_recons(dropout_tv)],'type','Spearman');
        rho = tmp(1,2);
        rho_all(i,m) = rho;

        tmp = corr([cell_cns(:) rc_recons(:)],'type','Spearman');
        rho1 = tmp(1,2);
        delta_rho_all(i,m) = rho1-rho2;

        sh1 = mean(silhouette(rc_recons,cell_assigns,'cosine'));
        delta_silhouette_all(i,m) = sh1-sh2;
        
        % coefficient of variation
        cvs_all_post = [];
        for c = 1:num_cell
            u_cns = unique(cell_cns(c,:));
            cvs = zeros(1,length(u_cns));
            for n = 1:length(u_cns)
                tv = cell_cns(c,:) == u_cns(n);
                cvs(n) = std(rc_recons(c,tv))/mean(rc_recons(c,tv));
            end
            cvs_all_post = [cvs_all_post cvs];
        end
        delta_cv_all(i,m) = mean(cvs_all_ori-cvs_all_post);
    end
end
 
save results_chisel.mat rho_all delta_rho_all delta_silhouette_all delta_cv_all;






