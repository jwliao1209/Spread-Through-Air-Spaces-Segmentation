function out = StatisticParse(Anno, Label)
    out.version   = Anno.version;
    out.size      = [Anno.imageHeight Anno.imageWidth];
    out.numTarget = numel(Anno.shapes);
    out.stasRatio = sum(Label(:)) / prod(out.size); 
end