#!/bin/bash

# F="dataset/F/F_S_r6200_c5400.bmp,dataset/F/F_T_r400_c400.bmp"
F="dataset/A/A_S_r1000_c1900.bmp,dataset/A/A_T_r35_c25.bmp"
  # dataset/B/B_S_r1000_c1600.bmp,dataset/B/B_T_r80_c70.bmp \
  # dataset/C/C_S_r600_c600.bmp,dataset/C/C_T_r90_c90.bmp \
  # dataset/D/D_S_r750_c1000.bmp,dataset/D/D_T_r55_c80.bmp \
  # dataset/E/E_S_r500_c1500.bmp,dataset/E/E_T_r66_c66.bmp"
BlockSize="16 32 64 128"
FixedBlockSize="32"
ThreadSize="4 8 16 32"
FixedThreadSize="32"
Method="SSD PCC"

Exec="./imgrep"
Result="./result.txt"

for f in $F
do
  IFS=',' read -ra files <<< "$f"
  for m in $Method
  do
    echo ${m} ${files[@]}
    # for block_size in $BlockSize
    # do
    #   echo "Benchmarking CUDA (b=${block_size} t=${FixedThreadSize})"
    #   ${Exec} -m ${m} -b ${block_size} -t ${FixedThreadSize} ${files[@]} 2>&1 | tee -a ${Result}
    #   echo | tee -a ${Result}
    # done
    for thread_size in $ThreadSize
    do
      echo "Benchmarking CUDA (b=${FixedBlockSize} t=${thread_size})"
      ${Exec} -m ${m} -b ${FixedBlockSize} -t ${thread_size} ${files[@]} 2>&1 | tee -a ${Result}
      echo | tee -a ${Result}
    done
    # echo "Benchmarking CPU"
    # ${Exec} -c -m ${m} ${files[@]} 2>&1 | tee -a ${Result}
    # echo | tee -a ${Result}
  done
done