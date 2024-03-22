timestamp=$(date +"%Y%m%d%H%M%S")
jobname="sbrnet_synthetic_data_gen_${timestamp}"


qsub -N "${jobname}" -o "/projectnb/tianlabdl/rjbona/sbrnet/.log/qsubs/${jobname}.qlog" /projectnb/tianlabdl/rjbona/sbrnet/sbrnet_core/scripts/gen_syn_data.qsub

