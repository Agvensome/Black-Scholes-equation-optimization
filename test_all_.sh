#!/bin/bash

option="bin"
mkdir -p ./$option

# make
mpicc -o ./${option}/bs_equation ./src/bs_equation.c -lm -Wall

option="Call"
mkdir -p ./$option

# mv
mv ./bin/bs_equation ./$option/bs_equation

cd ./$option

K=100.0
r=0.05
sigma=0.8
T=1.0
S_max=500.0
option_type=0
N=100
time_steps=50
save_interval=100
restart_file=""
current_time=0.0
stability_test=0


# 1. Numerical stability
echo "Numerical stability of explicit Eular"
mpirun -np 4 ./bs_equation -stability_test -explicit -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

echo "Numerical stability of implicit Eular"
mpirun -np 4 ./bs_equation -stability_test -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

# 2. Code verification
echo "Numerical stability of explicit Eular"
mpirun -np 4 ./bs_equation  -verification -explicit -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

echo "Numerical stability of implicit Eular"
mpirun -np 4 ./bs_equation  -verification -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

# 3. Parallelism
# fixed size scalability
echo "fixed size scalability of explicit Eular"
mpirun -np 4 ./bs_equation -explicit -fixed_size_test -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type
echo "fixed size scalability of implicit Eular"
mpirun -np 4 ./bs_equation -fixed_size_test -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

# isogranular scalability
echo "isogranular scalability of explicit Eular"
mpirun -np 4 ./bs_equation -explicit -isogranular_test -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type
echo "isogranular scalability of implicit Eular"
mpirun -np 4 ./bs_equation -isogranular_test -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

# 4. Postprocessing
echo "Postprocessing of explicit Eular"
mpirun -np 4 ./bs_equation  -post -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type
echo "Postprocessing of implicit Eular"
mpirun -np 4 ./bs_equation  -post -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

# archive
mkdir -p ./numerical_stability
mv stability_test_explicit.dat ./numerical_stability/stability_test_explicit.dat
mv stability_test_implicit.dat ./numerical_stability/stability_test_implicit.dat
mkdir -p ./code_verification
mv temporal_convergence_explicit.dat ./code_verification/temporal_convergence_explicit.dat
mv temporal_convergence_implicit.dat ./code_verification/temporal_convergence_implicit.dat
mv spatial_convergence_explicit.dat ./code_verification/spatial_convergence_explicit.dat
mv spatial_convergence_implicit.dat ./code_verification/spatial_convergence_implicit.dat
mkdir -p ./parallelism
mv fixed_size_scalability_explicit.dat ./parallelism/fixed_size_scalability_explicit.dat
mv fixed_size_scalability_implicit.dat ./parallelism/fixed_size_scalability_implicit.dat
mv isogranular_scalability_explicit.dat ./parallelism/isogranular_scalability_explicit.dat
mv isogranular_scalability_implicit.dat ./parallelism/isogranular_scalability_implicit.dat


cd ..

option="Put"
mkdir -p ./$option

# mv
mv ./Call/bs_equation ./$option/bs_equation

cd ./$option

K=100
r=0.02
sigma=0.8
T=2.0
S_max=300.0
option_type=1 
N=500
time_steps=1000
save_interval=100
restart_file=""
current_time=0.0
stability_test=0


# 1. Numerical stability
echo "Numerical stability of explicit Eular"
mpirun -np 4 ./bs_equation -stability_test -explicit -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

echo "Numerical stability of implicit Eular"
mpirun -np 4 ./bs_equation -stability_test -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

# 2. Code verification
echo "Numerical stability of explicit Eular"
mpirun -np 4 ./bs_equation  -verification -explicit -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

echo "Numerical stability of implicit Eular"
mpirun -np 4 ./bs_equation  -verification -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

# 3. Parallelism
# fixed size scalability
echo "fixed size scalability of explicit Eular"
mpirun -np 4 ./bs_equation -explicit -fixed_size_test -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type
echo "fixed size scalability of implicit Eular"
mpirun -np 4 ./bs_equation -fixed_size_test -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

# isogranular scalability
echo "isogranular scalability of explicit Eular"
mpirun -np 4 ./bs_equation -explicit -isogranular_test -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type
echo "isogranular scalability of implicit Eular"
mpirun -np 4 ./bs_equation -isogranular_test -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type

# 4. Postprocessing
echo "Postprocessing of explicit Eular"
mpirun -np 4 ./bs_equation  -post -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type
echo "Postprocessing of implicit Eular"
mpirun -np 4 ./bs_equation  -post -K $K -sigma $sigma -r $r -T $T -S_max $S_max -N $N -time_steps $time_steps -option_type $option_type


# archive
mkdir -p ./numerical_stability
mv stability_test_explicit.dat ./numerical_stability/stability_test_explicit.dat
mv stability_test_implicit.dat ./numerical_stability/stability_test_implicit.dat
mkdir -p ./code_verification
mv temporal_convergence_explicit.dat ./code_verification/temporal_convergence_explicit.dat
mv temporal_convergence_implicit.dat ./code_verification/temporal_convergence_implicit.dat
mv spatial_convergence_explicit.dat ./code_verification/spatial_convergence_explicit.dat
mv spatial_convergence_implicit.dat ./code_verification/spatial_convergence_implicit.dat
mkdir -p ./parallelism
mv fixed_size_scalability_explicit.dat ./parallelism/fixed_size_scalability_explicit.dat
mv fixed_size_scalability_implicit.dat ./parallelism/fixed_size_scalability_implicit.dat
mv isogranular_scalability_explicit.dat ./parallelism/isogranular_scalability_explicit.dat
mv isogranular_scalability_implicit.dat ./parallelism/isogranular_scalability_implicit.dat


cd ..

option="bin"

# mv
mv ./Put/bs_equation ./$option/bs_equation

# dat to txt
# 指定包含 .dat 文件的目录路径
source_directory="./"

# 遍历目录及其所有子目录中的所有 .dat 文件
find "$source_directory" -type f -name "*.dat" | while read -r file; do
    # 构造目标路径，将 .dat 替换为 .txt
    dest="${file%.dat}.txt"
    # 复制文件
    cp "$file" "$dest"
    echo "已复制 $file 为 $dest"
done