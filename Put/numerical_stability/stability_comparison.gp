

# ���ñ���ͱ�ǩ
set title "Black-Scholes ��ֵ�����ȶ��ԱȽ�"
set xlabel "��t / �����ȶ��Ա߽�"
set ylabel "�ȶ���״̬"
set yrange [-0.2:1.2]
set ytics ("�ȶ�" 0, "��ɢ" 1) nomirror
set key top left
set grid



# ������ʽ�����������ɫԲ��+���ߣ�
plot "stability_test_explicit.txt" using 3:4 with linespoints \
     pt 7 ps 1.5 lc rgb "red" lw 2 title "��ʽŷ������", \
     "stability_test_implicit.txt" using 3:4 with linespoints \
     pt 7 ps 1.5 lc rgb "blue" lw 2 title "��ʽŷ������"

# ���ͼ��˵��
set label 1 "��ʽŷ�������� ��t/���۱߽� > 1 ʱ��ɢ" at graph 0.6, graph 0.9 left tc rgb "red"
set label 2 "��ʽŷ�������ڲ��Է�Χ�ڱ����ȶ�" at graph 0.6, graph 0.8 left tc rgb "blue"




