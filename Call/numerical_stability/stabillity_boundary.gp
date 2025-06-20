set title "稳定性边界分析"
set xlabel "Δt / 理论稳定性边界"
set ylabel "时间步数"
# set logscale x
set logscale y
set key top right
set grid
set arrow from 1.22, graph 0 to 1.22, graph 1 nohead lc rgb "red" dt 2 lw 2

plot 'stability_test_explicit.txt' using 3:1 with points pt 7 ps 1.5 lc rgb "blue" title "显式方法 (稳定)", \
     'stability_test_explicit.txt' using ($4==1?$3:1/0):1 with points pt 6 ps 1.5 lc rgb "red" title "显式方法 (发散)", \
     'stability_test_implicit.txt' using ($4==1?$3:1/0):1 with points pt 4 ps 1.5 lc rgb "magenta" title "隐式方法 (发散)"