cmd=mpirun
total_time=$($cmd -n 3 ./sharded d b 0 g 2 | tail -n 1)
time=$(bc -l <<< "$total_time / 10000")

$cmd -n 3 ./sharded b

for i in {1..3}
do
  for load in 0.2 0.4 0.6 0.8 1
  do
    interval=$(bc -l <<< "$time / $load")

    for alg in "g" "gmin" "q" "c"
    do
      echo "d c $load $alg"
      $cmd -n 3 ./sharded d c $interval $alg 2
      echo "d p $load $alg"
      $cmd -n 3 ./sharded d p $interval $alg 2
    done

    for block_size in {25..300..25}
    do
      echo "s c $load $block_size"
      $cmd -n 3 ./sharded s c $interval $block_size 2
      echo "s p $load $block_size"
      $cmd -n 3 ./sharded s p $interval $block_size 2
    done
  done
done
