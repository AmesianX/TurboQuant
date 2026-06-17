#!/bin/bash
cd /home/user/work/TurboQuant
export DFLASH_DBG=1
build/bin/llama-server \
  -m /home/user/Models/DeepSeek-V4-Flash-GGUF/IQ2_XS-XL/DeepSeek-V4-Flash-IQ2_XS-XL-00001-of-00003.gguf \
  --model-draft /home/user/work/dflash-dsv4/dflash-drafter-tau163.gguf \
  --spec-type draft-dflash \
  -c 8192 -ngl 999 -ngld 999 -fa on -t 4 -np 1 -fit off --verbose \
  --host 127.0.0.1 --port 8090
