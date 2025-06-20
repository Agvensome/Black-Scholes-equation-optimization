set title "时间步长与稳定性关系"
set xlabel "时间步数"
set ylabel "Δt"
set logscale x
set logscale y
set key top right
set grid

plot 'stability_test_explicit.txt' using 1:2 with points pt 7 ps 1.5 lc rgb "blue" title "显式方法 (稳定)", \
     'stability_test_explicit.txt' using ($4==1?$1:1/0):2 with points pt 6 ps 1.5 lc rgb "red" title "显式方法 (发散)", \
     'stability_test_implicit.txt' using ($4==1?$1:1/0):2 with points pt 4 ps 1.5 lc rgb "magenta" title "隐式方法 (发散)"