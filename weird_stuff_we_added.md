ofs = 2, twice=in script and in pipeline
max_frames


todo:
1. decide on lora rank
2. maybe different lr
3. batch size
4. epochs
5. multi gpu


srun --pty --qos=normal --nodes=1 -G 1 -p mig --cpus-per-gpu=20  --container-mounts $PWD:"/workspace" --container-image="/rg/kimmel_prj/royve/prj/rev02_pytorchlightning+pytorch_lightning_video_pipe.sqsh" --container-save="/rg/kimmel_prj/royve/prj/rev03_pytorchlightning+pytorch_lightning_video_pipe.sqsh"  --container-workdir="/workspace" bash -i