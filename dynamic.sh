cmd=mpirun

echo 'benching'
# $cmd -n 3 ./sharded b

for i in {1..1}
do
  for load in 0.2 0.4 0.6 0.8 1
  do
    for alg in "g" "b" "ge"
    do
      echo "$load $alg"
      $cmd -n 3 ./sharded p $load s $alg
    done

    for block_size in {25..200..25}
    do
      echo "$load s $block_size"
      $cmd -n 3 ./sharded p $load s s $block_size gpu
    done
  done
done
