#!/bin/bash

# Copy contents of first file into all others
cat run_10_3_3.sh > run_10_4_4.sh
cat run_10_3_3.sh > run_10_5_5.sh
cat run_10_3_3.sh > run_11_3_3.sh
cat run_10_3_3.sh > run_11_4_4.sh
cat run_10_3_3.sh > run_11_5_5.sh
cat run_10_3_3.sh > run_12_3_3.sh
cat run_10_3_3.sh > run_12_4_4.sh
cat run_10_3_3.sh > run_12_5_5.sh
cat run_10_3_3.sh > run_13_3_3.sh
cat run_10_3_3.sh > run_13_4_4.sh
cat run_10_3_3.sh > run_13_5_5.sh
cat run_10_3_3.sh > run_14_3_3.sh
cat run_10_3_3.sh > run_14_4_4.sh
cat run_10_3_3.sh > run_14_5_5.sh
cat run_10_3_3.sh > run_15_3_3.sh
cat run_10_3_3.sh > run_15_4_4.sh
cat run_10_3_3.sh > run_15_5_5.sh

# Replace all numbers in files with the appropriate things
sed -i 's/10 3 3/10 4 4/g' run_10_4_4.sh; sed -i 's/10_3_3/10_4_4/g' run_10_4_4.sh
sed -i 's/10 3 3/10 5 5/g' run_10_5_5.sh; sed -i 's/10_3_3/10_5_5/g' run_10_5_5.sh
sed -i 's/10 3 3/11 3 3/g' run_11_3_3.sh; sed -i 's/10_3_3/11_3_3/g' run_11_3_3.sh
sed -i 's/10 3 3/11 4 4/g' run_11_4_4.sh; sed -i 's/10_3_3/11_4_4/g' run_11_4_4.sh
sed -i 's/10 3 3/11 5 5/g' run_11_5_5.sh; sed -i 's/10_3_3/11_5_5/g' run_11_5_5.sh
sed -i 's/10 3 3/12 3 3/g' run_12_3_3.sh; sed -i 's/10_3_3/12_3_3/g' run_12_3_3.sh
sed -i 's/10 3 3/12 4 4/g' run_12_4_4.sh; sed -i 's/10_3_3/12_4_4/g' run_12_4_4.sh
sed -i 's/10 3 3/12 5 5/g' run_12_5_5.sh; sed -i 's/10_3_3/12_5_5/g' run_12_5_5.sh
sed -i 's/10 3 3/13 3 3/g' run_13_3_3.sh; sed -i 's/10_3_3/13_3_3/g' run_13_3_3.sh
sed -i 's/10 3 3/13 4 4/g' run_13_4_4.sh; sed -i 's/10_3_3/13_4_4/g' run_13_4_4.sh
sed -i 's/10 3 3/13 5 5/g' run_13_5_5.sh; sed -i 's/10_3_3/13_5_5/g' run_13_5_5.sh
sed -i 's/10 3 3/14 3 3/g' run_14_3_3.sh; sed -i 's/10_3_3/14_3_3/g' run_14_3_3.sh
sed -i 's/10 3 3/14 4 4/g' run_14_4_4.sh; sed -i 's/10_3_3/14_4_4/g' run_14_4_4.sh
sed -i 's/10 3 3/14 5 5/g' run_14_5_5.sh; sed -i 's/10_3_3/14_5_5/g' run_14_5_5.sh
sed -i 's/10 3 3/15 3 3/g' run_15_3_3.sh; sed -i 's/10_3_3/15_3_3/g' run_15_3_3.sh
sed -i 's/10 3 3/15 4 4/g' run_15_4_4.sh; sed -i 's/10_3_3/15_4_4/g' run_15_4_4.sh
sed -i 's/10 3 3/15 5 5/g' run_15_5_5.sh; sed -i 's/10_3_3/15_5_5/g' run_15_5_5.sh