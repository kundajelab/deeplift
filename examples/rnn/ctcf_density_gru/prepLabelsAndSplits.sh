#!/usr/bin/env sh

echo $'id\tlabel' > labels.txt
zcat *pos*.gz | perl -lane 'if ($. > 1) {print("$F[0]\t1")}' >> labels.txt
zcat *neg*.gz | perl -lane 'if ($. > 1) {print("$F[0]\t0")}' >> labels.txt
mkdir splits
cat labels.txt | perl -lane 'if (($.%10 != 1) && ($.%10 != 2)) {print $_}' > splits/train.txt
cat labels.txt | perl -lane 'if ($.%10==1) {print $_}' > splits/valid.txt
cat labels.txt | perl -lane 'if ($.%10==2) {print $_}' > splits/test.txt
