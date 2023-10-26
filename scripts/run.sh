timestamp=$(date +"%Y%m%d%H%M%S")
jobname="sbrnet_training_${timestamp}"

qsub -N "${jobname}" -o "/projectnb/tianlabdl/jalido/sbrnet_proj/.log/qsubs/${jobname}.qlog" train.qsub
