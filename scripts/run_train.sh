archtecture = ${}
timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
jobname="sbrnet_training_${timestamp}"

qsub -N "${jobname}" -o "/projectnb/tianlabdl/nrabines/sbrnet/.log/qsubs/${jobname}.qlog" /projectnb/tianlabdl/nrabines/sbrnet/scripts/train.qsub
