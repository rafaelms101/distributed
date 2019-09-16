for i in {1..1}
do
        echo 'benching'
        mpirun -n 3 ./sharded b
        echo 'cpu'
        mpirun -n 3 ./sharded d c 0 c 2
        echo 'gpu'
        mpirun -n 3 ./sharded d c 0 gmin 2
        echo 'hybrid'
        mpirun -n 3 ./sharded d c 0 h 2
done
