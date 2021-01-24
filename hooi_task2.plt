set term pdf
set out "hooi_task2.pdf"
set xlabel "C" font "Arial-Italic,15"
set ylabel "R" font "Arial-Italic,15"
set xrange [-0.1:1.1]
set yrange [0:0.6]
set key right center font "Arial,10"
p "hooi_task2.dat" u 1:2:5 pt 6 ps 0.3 lt rgb 'red' w yerrorlines t "高次元直交反復を用いて近似した評価関数の勝率",\
"hooi_task2.dat" u 1:3:6 pt 2 ps 0.3 lt rgb 'blue' w yerrorlines t "高次元特異値分解を用いて近似した評価関数の勝率",\
