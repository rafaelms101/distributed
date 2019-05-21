for i in {1..1}
do
  #echo "Execution #$i"
  #echo "mpirun -n 3 ./sharded b"
  mpirun -n 3 ./sharded b

  for load in 0.2 0.4 0.6 0.8 1 1.2 2
  do
    # echo "mpirun -n 3 ./sharded d c $load min"
    # mpirun -n 3 ./sharded d c $load min
    # echo "mpirun -n 3 ./sharded d c $load max"
    # mpirun -n 3 ./sharded d c $load max
    #
    echo "d $load min"
    mpirun -n 3 ./sharded d p $load min
    echo "d $load max"
    mpirun -n 3 ./sharded d p $load max

    for block_size in {20..300..20}
    do
      # echo "mpirun -n 3 ./sharded s c $load $block_size"
      # mpirun -n 3 ./sharded s c $load $block_size
      echo "s $load $block_size"
      mpirun -n 3 ./sharded s p $load $block_size
    done
  done
done
