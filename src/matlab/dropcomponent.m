function  [drop_target, drop_target_list]= dropcomponent(pred, threshold)
if nargin==1
    threshold=0;
end
record = bwconncomp(pred);
PixelIdxList = record.PixelIdxList;
PredPixelsize = cellfun(@(x)length(x), PixelIdxList);
PredPixelbool = find(PredPixelsize < threshold);
drop_target = arrayfun(@(x) record.PixelIdxList{1,x}, ...
                       PredPixelbool, 'UniformOutput', false);
drop_target_list = cellfun(@(x) size(x,1), drop_target);
end


