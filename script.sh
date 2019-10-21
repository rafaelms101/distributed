nodes_max=8
nodes_step=2

export OMP_NUM_THREADS=7

git clone https://github.com/redmajor/distributed.git

cd distributed

git apply diff_folders

make sharded

# benching
echo 'benching'
lrun -N 1 -T 3 -c 10 -g 1  ./sharded b 

# now we compute the throughput of the CPU and GPU implementations
echo 'obtaining gpu throughput'
lrun -N 1 -T 3 -c 10 -g 1   ./sharded c 0 s b > gput
echo 'obtaining cpu throughput'
lrun -N 1 -T 3 -c 10 -g 1   ./sharded c 0 s c > cput

cpu_time=$(tail -1 cput)
gpu_time=$(tail -1 gput)

cpu_thr=$(echo "100000 / $cpu_time" | bc)
gpu_thr=$(echo "100000 / $gpu_time" | bc)

# now we "fix" things for running with the 1bi benchmark
cp prof/cpu_500000000_4096_8_10_16_5 prof/cpu_1000000000_4096_8_10_16_5
cp prof/gpu_500000000_4096_8_10_16_5 prof/gpu_1000000000_4096_8_10_16_5

git apply diff_1bi

make sharded

echo 'out-of-core nodes=1'
lrun -N 1 -T 4 -c 10 -g 1  ./sharded p 1 o h 1 2

for nodes in {2..$nodes_max..$nodes_step}
do
	echo "out-of-core nodes=$nodes"
	lrun -N $nodes -T 4 -c 10 -g 1  ./sharded p 1 o h 1 2
done

git restore config.h
git apply diff_folders

make sharded 

echo 'in-core nodes=1'
lrun -N 1 -T 4 -c 10 -g 1  ./sharded p 1 b $gpu_thr $cpu_thr b

for nodes in {2..$nodes_max..$nodes_step}
do
	echo "in-core nodes=$nodes"
	lrun -N $nodes -T 4 -c 10 -g 1  ./sharded p 1 b $gpu_thr $cpu_thr b
done
