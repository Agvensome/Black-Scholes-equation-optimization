set title "�ȶ��Ա߽����"
set xlabel "��t / �����ȶ��Ա߽�"
set ylabel "ʱ�䲽��"
# set logscale x
set logscale y
set key top right
set grid
set arrow from 1.22, graph 0 to 1.22, graph 1 nohead lc rgb "red" dt 2 lw 2

plot 'stability_test_explicit.txt' using 3:1 with points pt 7 ps 1.5 lc rgb "blue" title "��ʽ���� (�ȶ�)", \
     'stability_test_explicit.txt' using ($4==1?$3:1/0):1 with points pt 6 ps 1.5 lc rgb "red" title "��ʽ���� (��ɢ)", \
     'stability_test_implicit.txt' using ($4==1?$3:1/0):1 with points pt 4 ps 1.5 lc rgb "magenta" title "��ʽ���� (��ɢ)"