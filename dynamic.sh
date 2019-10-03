cmd=mpirun

echo 'benching'
$cmd -n 3 ./sharded b

for i in {1..1}
do
  for load in 0.2 0.4 0.6 0.8 1
  do
    for alg in "q" "gmin" "g"
    do
      echo "d c $load $alg"
      $cmd -n 3 ./sharded d c $load $alg 2
      echo "d p $load $alg"
      $cmd -n 3 ./sharded d p $load $alg 2
    done

    for block_size in {25..200..25}
    do
      echo "s c $load $block_size"
      $cmd -n 3 ./sharded s c $load $block_size cpu 2
      echo "s p $load $block_size"
      $cmd -n 3 ./sharded s p $load $block_size cpu 2
    done
  done
done
