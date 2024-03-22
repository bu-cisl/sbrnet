timestamp=$(date +"%Y%m%d%H%M%S")
jobname="sbrnet_training_${timestamp}"

qsub -N "${jobname}" -o "/projectnb/tianlabdl/rjbona/sbrnet/.log/qsubs/${jobname}.qlog" /projectnb/tianlabdl/rjbona/sbrnet/sbrnet_core/sbrnet/scripts/train.qsub
