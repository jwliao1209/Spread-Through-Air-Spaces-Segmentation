function img_fillg = detection(img_loc, filename)
img = rgb2gray(imread(fullfile(img_loc, [filename(1:end-4) '.jpg'])));
img_e = edge(img, 'sobel');
img_ed = imclosen(img_e, 9, 1);
img_fill = imfill(img_ed, 'holes');

[target, ~] = dropcomponent12(img_fill, 2500);
img_fillg = img_fill;
for i = 1:length(target)
    idx = target{1,i};
    img_fillg(idx)=0;
end
end