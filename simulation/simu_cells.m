function cell_assignments = simu_cells(num_cell,num_clone)

if num_cell < num_clone+1
    disp('number of cells should be larger than number of clones');
    cell_assignments = [];
else
    n = floor(num_cell*0.5);
    m = max(1,floor(n/(num_clone+1)));
    cell_assignments = repmat(1:num_clone+1,1,m);
    
    weights = rand(1,num_clone+1);
    tmp = randsrc(1,num_cell-(num_clone+1)*m,[1:num_clone+1; weights/sum(weights)]);
    cell_assignments = [cell_assignments tmp];
end

end