clc;
clear;
fclose all;

base_dir = './denoising_rc';
method_names = {'scImpute' 'DeepImpute' 'GE-Impute' 'SCDD' 'scMultiGAN' 'TsImpute' 'MAGIC' 'bayNorm' 'CarDEC' 'scTCA'};
num_tree = 5;
num_cells = [500 1000];
num_clones = [7 9 11];
num_bins = [12000 18000 24000];

rho_all = cell(length(num_cells),length(num_clones));
delta_rho_all = cell(length(num_cells),length(num_clones));
delta_silhouette_all = cell(length(num_cells),length(num_clones));
delta_cv_all = cell(length(num_cells),length(num_clones));

for i = 1:length(num_cells)
    num_cell = num_cells(i);
    for j = 1:length(num_clones)
        num_clone = num_clones(j);
        rhos = zeros(length(num_bins),length(method_names),num_tree);
        delta_rhos = zeros(length(num_bins),length(method_names),num_tree);
        delta_silhouettes = zeros(length(num_bins),length(method_names),num_tree);
        delta_cvs = zeros(length(num_bins),length(method_names),num_tree);
        for k = 1:length(num_bins)
            num_bin = num_bins(k);
            for t = 1:num_tree
                ds = ['ploidy_2_tree_' num2str(t) '_clones_' num2str(num_clone) '_bins_' num2str(num_bin)];
                disp(['processing ' ds '_cells_' num2str(num_cell)])
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
                dropout_tv = rc_ori == 0;
                
                for c = 1:num_cell
                    ploidy = mode(cell_cns(c,:));
                    tv = cell_cns(c,:) == ploidy;
                    rc = mean(rc_ori(c,tv));
                    rc_ori(c,:) = rc_ori(c,:)/(2*rc/ploidy+eps);
                end
                
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
                cv1 = mean(cvs_all_ori);
                
                clear rc_ori;
                for m = 1:length(method_names)
                    result_dir = [base_dir '/results/simu/'];
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
                    
                    tmp = corr([cell_cns(dropout_tv) rc_recons(dropout_tv)],'type','Spearman');
                    rho = tmp(1,2);
                    rhos(k,m,t) = rho;
                    
                    tmp = corr([cell_cns(:) rc_recons(:)],'type','Spearman');
                    rho1 = tmp(1,2);
                    delta_rhos(k,m,t) = rho1-rho2;
                    
                    sh1 = mean(silhouette(rc_recons,cell_assigns,'cosine'));
                    delta_silhouettes(k,m,t) = sh1-sh2;
                    
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
                    cv2 = mean(cvs_all_post);
                    delta_cvs(k,m,t) = mean(cvs_all_ori-cvs_all_post);
                end
            end
        end
        rho_all{i,j} = rhos;
        delta_rho_all{i,j} = delta_rhos;
        delta_silhouette_all{i,j} = delta_silhouettes;
        delta_cv_all{i,j} = delta_cvs;
    end
end
 
save results_simu.mat rho_all delta_rho_all delta_silhouette_all delta_cv_all;


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




