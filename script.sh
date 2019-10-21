export OMP_NUM_THREADS=7

git clone https://github.com/redmajor/distributed.git

cd distributed

git apply diff_folders

make sharded

# benching
lrun -N 1 -T 3 -c 10 -g 1  ./sharded b 

# now we "fix" things for running with the 1bi benchmark
cp cpu_500000000_4096_8_10_16_5 cpu_1000000000_4096_8_10_16_5
cp gpu_500000000_4096_8_10_16_5 gpu_1000000000_4096_8_10_16_5

git apply diff_1bi

make sharded

echo 'out-of-core nodes=1'
lrun -N 1 -T 4 -c 10 -g 1  ./sharded p 1 o h 1 2

for nodes in {2..8..2}
do
	echo 'out-of-core nodes=$nodes'
	lrun -N $nodes -T 4 -c 10 -g 1  ./sharded p 1 o h 1 2
done

git restore config.h
git apply diff_folders

make sharded 

echo 'in-core nodes=1'
lrun -N 1 -T 4 -c 10 -g 1  ./sharded p 1 o h 1 2

for nodes in {2..8..2}
do
	echo 'in-core nodes=$nodes'
	lrun -N $nodes -T 4 -c 10 -g 1  ./sharded p 1 o h 1 2
done
