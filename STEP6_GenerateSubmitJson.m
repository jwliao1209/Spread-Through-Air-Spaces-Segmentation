clear; clc; close all;
addpath(genpath(fullfile('src', 'matlab')));

%% setup
JsonRoot    = fullfile('Json_Data');
DatasetRoot = fullfile('Dataset');
PublicRoot  = fullfile(DatasetRoot, 'Public_Image');
PrivateRoot = fullfile(DatasetRoot, 'Private_Image');

Process = @(X) struct('filename', replace(X.name, '.jpg', ''));

%% generate json file
if exist(PublicRoot, 'dir')
    Content = dir(fullfile(PublicRoot, '*.jpg'));
    Content = arrayfun(Process, Content, 'UniformOutput', false);
    WriteJson(fullfile(JsonRoot, 'Public.json'), Content);
end

if exist(PrivateRoot, 'dir')
    Content = dir(fullfile(PrivateRoot, '*.jpg'));
    Content = arrayfun(Process, Content, 'UniformOutput', false);
    WriteJson(fullfile(JsonRoot, 'Private.json'), Content);
end