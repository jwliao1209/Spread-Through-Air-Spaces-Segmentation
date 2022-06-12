function pred = rrdd(pred, num, times)
if ~exist('num', 'var')
    num = 5;
end
if ~exist('times', 'var')
    times = 2;
end

SE = strel('square', num);
pred = 255-double(pred);
for j = 1:times
    pred = imerode(pred, SE);
end
for j = 1:times
    pred = imdilate(pred, SE);
end
pred = 255-double(pred);
end