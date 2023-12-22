timestamp=$(date +"%Y%m%d%H%M%S")
jobname="sbrnet_array_training_${timestamp}"

qsub -N "${jobname}" -o "/projectnb/tianlabdl/jalido/sbrnet_proj/.log/qsubs/${jobname}.qlog" /projectnb/tianlabdl/jalido/sbrnet_proj/scripts/array_train.qsub
