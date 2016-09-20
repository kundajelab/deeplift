#!/usr/bin/env sh

zcat EmptyBackground_seqLength-250_numSeqs-2000.simdata.gz | perl -lane 'if ($. == 1) {print $_} else {$seq = $F[1]; $seq =~ s/./N/g; print "allNs_$F[0]\t$seq\t\t"'} | gzip -c > AllNs.gz
echo $'id\tgata\ttal\tgataAndTal' > labels.txt
#zcat EmptyBackground*.gz | perl -lane 'if ($. > 1) {print("$F[0]\t0\t0\t0")}' >> labels.txt
zcat AllNs*.gz | perl -lane 'if ($. > 1) {print("$F[0]\t0\t0\t0")}' >> labels.txt
zcat *gataOnly*.gz | perl -lane 'if ($. > 1) {print("$F[0]\t1\t0\t0")}' >> labels.txt
zcat *talOnly*.gz | perl -lane 'if ($. > 1) {print("$F[0]\t0\t1\t0")}' >> labels.txt
zcat *gataAndTal*.gz | perl -lane 'if ($. > 1) {print("$F[0]\t1\t1\t1")}' >> labels.txt
mkdir splits
cat labels.txt | perl -lane 'if ($. > 1 && (($.%10 != 1) && ($.%10 != 2))) {print $_}' > splits/train.txt
cat labels.txt | perl -lane 'if ($. > 1 && ($.%10==1)) {print $_}' > splits/valid.txt
cat labels.txt | perl -lane 'if ($. > 1 && ($.%10==2)) {print $_}' > splits/test.txt
