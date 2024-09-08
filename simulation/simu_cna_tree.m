function [tree,cna_events,cn_profile] = simu_cna_tree(num_clone,ploidy,segs,lambda_r,lambda_c,max_cn)

num_seg = size(segs,1);

if num_clone <= 0
    error('number of subclones should be a positive integer');
end

% simulate subclonal tree
tree = zeros(1,num_clone+1);
tree(2) = 1;
for n = 3:num_clone+1
    k = randperm(n-2, 1);
    tree(n) = k+1;
end

% simulate CNA events
while 1
    cn_profile = zeros(num_clone+1,num_seg);
    cn_profile(1,:) = ploidy;
    cna_events = zeros(num_clone+1,num_seg);
    for n = 2:num_clone+1
        p = tree(n);
        events_p = cna_events(p,:);
        profile_p = cn_profile(p,:);
        flag = 0;
        while flag == 0
            num_seg_change = poissrnd(lambda_r,1)+1;
            while num_seg_change > num_seg
                num_seg_change = poissrnd(lambda_r,1)+1;
            end
            selected_segs = sort(randperm(num_seg,num_seg_change));
            tv = ismember(1:num_seg,selected_segs);
            num_copy = poissrnd(lambda_c,1,num_seg_change)+1;
            sign = randsrc(1,num_seg_change,[1 -1;0.5 0.5]);

            if any(profile_p(selected_segs)==0)
                continue;
            end
            profile = profile_p;
            profile(selected_segs) = profile(selected_segs)+num_copy.*sign;
            if any(profile < 1) || any(profile > max_cn) % reject the events
                continue;
            end

            if all(events_p(tv) ~= 0) && all(events_p(~tv) == 0)
                tmp = events_p(tv)+num_copy.*sign; 
                if any(tmp ~= 0) % make sure events of the parent node are not completely cancelled
                    flag = 1;
                end
            else
                flag = 1;
            end
        end
        cna_events(n,selected_segs) = num_copy.*sign;
        cn_profile(n,:) = profile_p+cna_events(n,:);
    end
    
    for n = 1:num_clone
        for m = n+1:num_clone+1
            if sum(cn_profile(n,:)==cn_profile(m,:)) == num_seg % two clones have same copy number profiles
                continue;
            end
        end
    end

    % calculte ploidy
    acns = zeros(1,num_clone+1);
    prob_ploidy = zeros(1,num_clone+1);
    acns(1) = ploidy;
    prob_ploidy(1) = 1;
    for n = 2:num_clone+1
        acns(n) = cn_profile(n,:)*segs(:,end)/sum(segs(:,end));
        tv = cn_profile(n,:) == ploidy;
        prob_ploidy(n) = sum(segs(tv,end))/sum(segs(:,end));
    end
    if all(acns > ploidy-0.2 & acns < ploidy+0.2) && all(prob_ploidy > 0.5)
        break;
    end
end

end