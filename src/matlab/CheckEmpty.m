function CheckEmpty(ImgRoot)
    % Given a prediction folder, check how many of prediction are empty
    % inside
    FileList = dir(fullfile(ImgRoot, '*.png'));
    if ~numel(FileList)
        error('There is no file in such location that you have given. Please check again.\n');
    end
    
    EmptyCount = 0;
    for Info = FileList'
        FileName = Info.name;
        Pred = double(imread(fullfile(ImgRoot, FileName)));
        Pred = Pred(:,:,1);
        nz = nnz(Pred);

        if ~nz
            EmptyCount = EmptyCount + 1;
        end
        fprintf('Compare %s  nz:%d.\n', FileName, nz);
    end
    fprintf('Number of empty prediction:%2d.\n', EmptyCount);
end