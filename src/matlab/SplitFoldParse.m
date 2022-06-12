function out = SplitFoldParse(FileName, Anno, Label)
    if isa(Anno.shapes, 'struct')
        GET_POINTS = @(X) X.points;
    else
        GET_POINTS = @(X) X{1}.points;
    end
    POINTS2BBOX   = @(X) [min(GET_POINTS(X)) max(GET_POINTS(X))];
    
    out.filename  = FileName;
    out.version   = Anno.version;
    out.height    = Anno.imageHeight;
    out.width     = Anno.imageWidth;
    out.numobj    = numel(Anno.shapes);
    out.bbox_xyxy = arrayfun(POINTS2BBOX, Anno.shapes, ...
                             'UniformOutput', false);
    out.ratio     = sum(Label(:)) / numel(Label);
end