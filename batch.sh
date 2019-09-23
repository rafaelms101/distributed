cmd=mpirun

declare -A pairs=([1000]=0.7 [2000]=0.7 [2500]=0.68 [5000]=0.64 [10000]=0.6 [25000]=0.6 [50000]=0.54)
for i in {1..3}
do
	for bs in "${!pairs[@]}"
	do
	  ratio=${pairs[$bs]}

	  for alg in "cpu" "gpu" "h $ratio"
	  do
	  	echo "$bs $alg"
	  	$cmd -n 3 ./sharded s b 0 $bs $alg 2
	  done
	done
done