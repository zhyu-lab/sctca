function segs_all = generate_segs(seg_size_range,seg_size_weight,num_bin,bin_size)

min_seg_size = min(seg_size_range);

seq_len = num_bin*bin_size;

segs_all = [];
flag = 0;
while flag == 0
    spos = 1;
    segs = [];
    while spos <= seq_len-2*min_seg_size*1000000
        seg_size = randsrc(1,1,[seg_size_range;seg_size_weight/sum(seg_size_weight)]);
        seg_size = seg_size*1000000;
        epos = spos+seg_size-1;
        if epos <= seq_len
            segs = [segs; spos epos seg_size];
            spos = epos+1;
        end
    end
    if spos <= seq_len
        seg_size = seq_len-spos+1;
        if seg_size >= min_seg_size*1000000
            segs = [segs; spos seq_len seg_size];
            flag = 1;
        end
    end
end
segs_all = [segs_all; segs];


end