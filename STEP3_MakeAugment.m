clear; clc; close all;
addpath(genpath(fullfile('src', 'matlab')));

%% setup
mode     = 'Train'; % Train, Public
formList = {'norm', 'Himg', 'Eimg'};

DatasetRoot = fullfile('Dataset');
SrcRoot     = fullfile(DatasetRoot, [mode '_Images']);
DestRoot    = @(mode, form) fullfile(DatasetRoot, [mode '_' form]);
for k = 1 : numel(formList)
    form = formList{k};
    if ~exist(DestRoot(mode, form), 'dir')
        mkdir(DestRoot(mode, form));
    end
end

%% produce augment data
ImgList = dir(fullfile(SrcRoot, '*.jpg'));

tic;
for Info = ImgList'
    FileName = Info.name;
    fprintf(['Processing ' FileName '\n']);

    Img = imread(fullfile(SrcRoot, FileName));
    [Inorm, H, E] = normalizeStaining(Img);

    imwrite(Inorm, fullfile(DestRoot(mode, formList{1}), FileName));
    imwrite(H, fullfile(DestRoot(mode, formList{2}), FileName));
    imwrite(E, fullfile(DestRoot(mode, formList{3}), FileName));
end
toc;