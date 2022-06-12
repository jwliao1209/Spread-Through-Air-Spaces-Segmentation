clear; clc; close all;
addpath(genpath(fullfile('src', 'matlab')));

%% Setup path
DatasetRoot = fullfile('Dataset');
AnnoRoot    = fullfile(DatasetRoot, 'Train_Annotations');
LabelRoot   = fullfile(DatasetRoot, 'Train_Labels');

AnnoList = dir(fullfile(AnnoRoot, '*.json'));

%% Loop over the data
% loop backward such that preallocate memory in the first iteration
for k = numel(AnnoList): -1 : 1
    Info = AnnoList(k);
    FileName = replace(Info.name, '.json', '');
    Anno  = ReadJson(fullfile(AnnoRoot, [FileName '.json']));
    Label = imread(fullfile(LabelRoot, [FileName '.png']));

    Stat(k,1) = StatisticParse(Anno, Label);
end
Stat = struct2table(Stat);

%% Show the result
% image version count
[VerCount, VerType] = groupcounts(Stat.version);
PlotCountBar(VerType, VerCount, "ver. type", "num of data", "version count");

% image size count
[SizeCount, SizeType] = groupcounts(Stat.size);
heightType = arrayfun(@(x) num2str(x), SizeType{:,1}, 'UniformOutput', false);
widthType  = arrayfun(@(x) num2str(x), SizeType{:,2}, 'UniformOutput', false);
SizeType = cellfun(@(h,w) ['(' h ',' w ')'], heightType, widthType, ...
                   'UniformOutput', false);
PlotCountBar(SizeType, SizeCount, "size type", "num of data", "size count");

% num of target in one image count
[NTCount, NTType] = groupcounts(Stat.numTarget);
PlotCountBar(NTType, NTCount, "num of target", "num of data", "num of target count");

% STAS ratio histogram
figure(4);
histogram(Stat.stasRatio);
xlabel("stas ratio");
ylabel("num of data");
title("histogram for stats ratio");
