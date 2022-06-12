function pred = imclosen(pred, num, times)
if ~exist('num', 'var')
    num = 5;
end
if ~exist('times', 'var')
    times = 2;
end

SE = strel('square', num);
for j = 1:times
    pred = imdilate(pred, SE);
end
for j = 1:times
    pred = imerode(pred, SE);
end
end