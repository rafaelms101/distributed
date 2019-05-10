for i in {1..10}
do
  echo "Execution #$i"
  echo "mpirun -n 3 ./sharded b"
  mpirun -n 3 ./sharded b

  for load in 0.2 0.4 0.6 0.8 1
  do
    for block_size in 10 20 30 40 50
    do
      echo "mpirun -n 3 ./sharded s c $load $block_size"
      mpirun -n 3 ./sharded s c $load $block_size
    done

    echo "mpirun -n 3 ./sharded d c $load min"
    mpirun -n 3 ./sharded d c $load min
    echo "mpirun -n 3 ./sharded d c $load max"
    mpirun -n 3 ./sharded d c $load max
  done

  for block_size in 10 20 30 40 50
  do
    echo "mpirun -n 3 ./sharded s p 0.5 $block_size"
    mpirun -n 3 ./sharded s p 0.5 $block_size
  done

  echo "mpirun -n 3 ./sharded d p 0.5 min"
  mpirun -n 3 ./sharded d p 0.5 min
  echo "mpirun -n 3 ./sharded d p 0.5 max"
  mpirun -n 3 ./sharded d p 0.5 max
done
