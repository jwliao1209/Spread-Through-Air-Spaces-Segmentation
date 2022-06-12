function pred_fillg = Selection(pred_d, lower_thres, high_thres, drop_thres, ratio_thres)
% filter out those pixel with
% (confidence score < drop_thres)
pred_d(pred_d < drop_thres) = 0;

% find the connected component
record = bwconncomp(pred_d);

% filter out those connected component with 
% (all confidence score < high_thres)
PredPixelmax = cellfun(@(x) max(pred_d(x), [], 'all'), record.PixelIdxList);
PredPixelbool = find(PredPixelmax > high_thres);
goal_target = arrayfun(@(x) record.PixelIdxList{1,x}, PredPixelbool, 'UniformOutput', false);

pred_fillg = zeros(size(pred_d));
LL = length(goal_target);

for i = 1:LL
    idx = goal_target{1,i};

    ratio = sum(pred_d(idx) > high_thres*0.9) / length(idx);
    idx = idx(pred_d(idx) > lower_thres);

    if  ratio >= ratio_thres
        pred_fillg(idx) = 1;
    else
        pred_fillg(idx) = 0;
    end
end
end