cmd=mpirun

echo 'cpu benching'
./simple cpu 5 1000 5 > cpu_bench
echo 'gpu benching'
./simple gpu 5 1000 5 > gpu_bench

for i in {1..5}
do
  for load in 0.2 0.4 0.6 0.8 1 1.2
  do
    for alg in "c" "g 1 4" "g 1 2"  "g 3 4" "f 1 4" "f 1 2"  "f 3 4" "h 1 4" "h 1 2"  "h 3 4" 
    do
      echo "c $load $alg"
      $cmd -n 3 ./sharded c $load 2 $alg
      echo "p $load $alg"
      $cmd -n 3 ./sharded p $load 2 $alg
    done
  done
done
