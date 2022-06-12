function pred = imagefill(pred, fillnum)
[row, col] = size(pred);
if ~exist('fillnum', 'var')
    fillnum=8;
end
if sum(pred(:,1)) % left edge
    pred = cat(2, ones(row,1), pred);
    pred = imfill(double(pred),fillnum);
    pred = pred(:, 2:end);

elseif sum(pred(:,end)) % right edge
    pred = cat(2, pred, ones(row,1));
    pred = imfill(double(pred),fillnum);
    pred = pred(:, 1:end-1);

elseif sum(pred(1,:)) % top edge
    pred = cat(1, ones(1, col), pred);
    pred = imfill(double(pred),fillnum);
    pred = pred(2:end,:);

elseif sum(pred(end,:)) % buttom edge
    pred = cat(1, pred, ones(1, col));
    pred = imfill(double(pred),fillnum);
    pred = pred(1:end-1,:);
end
pred = imfill(double(pred),fillnum);
end