function  [drop_target, drop_target_list]= dropcomponent12(pred, threshold, threshold1)
% pretain the componnent
if nargin==1
    threshold=0;
    threshold1=1e5;
elseif nargin==2
    threshold1=1e5;
end
record = bwconncomp(pred);
PixelIdxList = record.PixelIdxList;
PredPixelctr = cellfun(@(x)length(x), PixelIdxList);
PredPixelbool = find(PredPixelctr<threshold);
PredPixelbool1 = find(PredPixelctr>threshold1);
drop_target1 = arrayfun(@(x) record.PixelIdxList{1,x}, ...
    PredPixelbool, 'UniformOutput', false);
drop_target2 = arrayfun(@(x) record.PixelIdxList{1,x}, ...
    PredPixelbool1, 'UniformOutput', false);
drop_target = [drop_target1(:)', drop_target2(:)'];

drop_target_list1 = cellfun(@(x) size(x,1), drop_target);
drop_target_list2 = cellfun(@(x) size(x,1), drop_target1);
drop_target_list = [drop_target_list1 drop_target_list2];
end


