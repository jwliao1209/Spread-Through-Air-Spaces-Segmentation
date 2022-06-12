function  [target, target_list]= component(pred)
record = bwconncomp(pred);
PixelIdxList = record.PixelIdxList;
PredPixelctr = cellfun(@(x)length(x), PixelIdxList);
PredPixelbool = find(PredPixelctr);
target = arrayfun(@(x) record.PixelIdxList{1,x}, ...
    PredPixelbool, 'UniformOutput', false);
target_list = cellfun(@(x) size(x,1), target);
end


