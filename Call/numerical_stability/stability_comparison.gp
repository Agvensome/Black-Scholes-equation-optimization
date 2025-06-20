

# 设置标题和标签
set title "Black-Scholes 数值方法稳定性比较"
set xlabel "Δt / 理论稳定性边界"
set ylabel "稳定性状态"
set yrange [-0.2:1.2]
set ytics ("稳定" 0, "发散" 1) nomirror
set key top left
set grid



# 绘制显式方法结果（红色圆点+折线）
plot "stability_test_explicit.txt" using 3:4 with linespoints \
     pt 7 ps 1.5 lc rgb "red" lw 2 title "显式欧拉方法", \
     "stability_test_implicit.txt" using 3:4 with linespoints \
     pt 7 ps 1.5 lc rgb "blue" lw 2 title "隐式欧拉方法"

# 添加图例说明
set label 1 "显式欧拉方法在 Δt/理论边界 > 1 时发散" at graph 0.6, graph 0.9 left tc rgb "red"
set label 2 "隐式欧拉方法在测试范围内保持稳定" at graph 0.6, graph 0.8 left tc rgb "blue"




