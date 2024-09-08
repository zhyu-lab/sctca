clc;
clear;
fclose all;

% ploidy
ploidy = 2;

% sequencing related paras
read_len = 84;
coverage_range = [0.02 0.05];
bin_size = 10000; % candidates:10000 20000
num_cells = [500 1000];
num_bins = [6000 9000 12000]*20000/bin_size;
num_clones = [7 9 11];
num_tree = 5;

if bin_size == 10000
    fid = fopen('dropout_rates_10k.txt','r');
    line = fgetl(fid);
    dropout_rates = str2double(regexp(line,',','split'));
    fclose(fid);
elseif bin_size == 20000
    fid = fopen('dropout_rates_20k.txt','r');
    line = fgetl(fid);
    dropout_rates = str2double(regexp(line,',','split'));
    fclose(fid);
else
    dropout_rates = [0.1];
end

% amplification bias
bias_mean = 1;
bias_lower_limit = 0;
bias_upper_limit = 2;
bias_sd = (bias_upper_limit-bias_mean)/3;
untruncated_bias_dist = makedist('Normal',bias_mean,bias_sd);
truncated_bias_dist = truncate(untruncated_bias_dist,bias_lower_limit,bias_upper_limit);

% copy number paras
max_cn = 10;
seg_size_range = [0.2:0.4:3.0 4:2:18 20:20:100];
seg_size_weight = ones(1,length(seg_size_range));
seg_size_weight(seg_size_range >= 20) = 0.5;
seg_size_weight(seg_size_range <= 1) = 0.5;
lambda_r = 3;
lambda_c = 1;

% save results
output_dir = './results/';
mkdir(output_dir);

coverage_mean = mean(coverage_range);
coverage_std = (coverage_mean-coverage_range(1))/3;
untruncated_dist = makedist('Normal',coverage_mean,coverage_std);
truncated_dist = truncate(untruncated_dist,coverage_range(1),coverage_range(2));

for num_clone = num_clones
    for i = 1:num_tree
        for num_bin = num_bins
            while 1
                segs = generate_segs(seg_size_range,seg_size_weight,num_bin,bin_size);
                if size(segs,1) > 10
                    break;
                end
            end
            [tree,cna_events,cn_profile] = simu_cna_tree(num_clone,ploidy,segs,lambda_r,lambda_c,max_cn);
            fn = ['ploidy_' num2str(ploidy) '_tree_' num2str(i) '_clones_' num2str(num_clone) '_bins_' num2str(num_bin) '.tree'];
            fid = fopen([output_dir fn],'w');
            for k = 1:length(tree)
                if k < length(tree)
                    fprintf(fid,'%d\t',tree(k));
                else
                    fprintf(fid,'%d\n',tree(k));
                end
            end
            
            cn_profile_bins = [];
            for k = 1:size(segs,1)
                bin_count = round(segs(k,3)/bin_size);
                cn_profile_bins = [cn_profile_bins repmat(cn_profile(:,k),1,bin_count)];
            end
            
            cn_profile_bins(1,:) = 2;
            fn = ['ploidy_' num2str(ploidy) '_tree_' num2str(i) '_clones_' num2str(num_clone) '_bins_' num2str(num_bin) '.cn'];
            fid = fopen([output_dir fn],'w');
            bin_spos = 1;
            for k = 1:num_bin-1
                fprintf(fid,'%d-%d,',bin_spos,bin_spos+bin_size-1);
                bin_spos = bin_spos+bin_size;
            end
            fprintf(fid,'%d-%d\n',bin_spos,bin_spos+bin_size-1);
            for k = 1:size(cn_profile_bins,1)
                for j = 1:size(cn_profile_bins,2)-1
                    fprintf(fid,'%d,',cn_profile_bins(k,j));
                end
                fprintf(fid,'%d\n',cn_profile_bins(k,end));
            end
            fclose(fid);
            
            for num_cell = num_cells
                cell_assignments = simu_cells(num_cell,num_clone);
                fn = ['ploidy_' num2str(ploidy) '_tree_' num2str(i) '_clones_' num2str(num_clone) '_bins_' num2str(num_bin) '_cells_' num2str(num_cell) '.cell_assigns'];
                fid = fopen([output_dir fn],'w');
                for k = 1:length(cell_assignments)-1
                    fprintf(fid,'%d\t',cell_assignments(k));
                end
                fprintf(fid,'%d\n',cell_assignments(end));
                fclose(fid);
                
                fn = ['ploidy_' num2str(ploidy) '_tree_' num2str(i) '_clones_' num2str(num_clone) '_bins_' num2str(num_bin) '_cells_' num2str(num_cell) '.rc'];
                fid = fopen([output_dir fn],'w');
                coverages_cell = random(truncated_dist,1,num_cell);
                dropout_rates_cell = randsrc(1,num_cell,[dropout_rates; ones(1,length(dropout_rates))/length(dropout_rates)]);
                for c = 1:num_cell
                    k = cell_assignments(c);
                    cn_cell = cn_profile_bins(k,:);
                    read_count = round(coverages_cell(c)*num_bin*bin_size/read_len);
                    bias_factors = random(truncated_bias_dist,1,num_bin);
                    rindxs = randperm(num_bin,floor(num_bin*dropout_rates_cell(c)));
                    bias_factors(rindxs) = 0;
                    weights = cn_cell.*bias_factors;
                    sampled_bins = randsrc(1,read_count,[1:num_bin; weights/sum(weights)]);
                    read_counts = hist(sampled_bins,1:num_bin);
                    for m = 1:num_bin-1
                        fprintf(fid,'%d,',read_counts(m));
                    end
                    fprintf(fid,'%d\n',read_counts(end));
                end
                fclose(fid);
            end
        end
    end
end














