timestamp=$(date +"%Y%m%d%H%M%S")
jobname="sbrnet_synthetic_data_gen_${timestamp}"


qsub -N "${jobname}" -o "/projectnb/tianlabdl/jalido/sbrnet_proj/.log/qsubs/${jobname}.qlog" /projectnb/tianlabdl/jalido/sbrnet_proj/scripts/gen_syn_data.qsub
