set title "ʱ�䲽�����ȶ��Թ�ϵ"
set xlabel "ʱ�䲽��"
set ylabel "��t"
set logscale x
set logscale y
set key top right
set grid

plot 'stability_test_explicit.txt' using 1:2 with points pt 7 ps 1.5 lc rgb "blue" title "��ʽ���� (�ȶ�)", \
     'stability_test_explicit.txt' using ($4==1?$1:1/0):2 with points pt 6 ps 1.5 lc rgb "red" title "��ʽ���� (��ɢ)", \
     'stability_test_implicit.txt' using ($4==1?$1:1/0):2 with points pt 4 ps 1.5 lc rgb "magenta" title "��ʽ���� (��ɢ)"