function PlotCountBar(Type, Count, XLabel, YLabel, Title)
    figure();
    bar(categorical(Type), Count);
    xlabel(XLabel);
    ylabel(YLabel);
    title(Title);
end