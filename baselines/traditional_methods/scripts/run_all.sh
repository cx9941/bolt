set -o errexit
for backbone in bert
do
for seed in 0 1 2 3 4
do
for dataset_name in mcid reuters
do
for rate in 0.25 0.5 0.75
do
sh scripts/run.sh $dataset_name $rate $backbone $seed 1
done
done
done
done

# set -o errexit
# for backbone in bert
# do
# for seed in $1
# do
# for dataset_name in stackoverflow
# do
# for rate in 0.75 0.5 0.25
# do
# sh scripts/run.sh $dataset_name $rate $backbone $seed $2
# done
# done
# done
# done