#!/usr/bin/env bash
mkdir splits
zcat labels.gz | perl -lane 'if ($. > 1 && (($.%10 != 1) && ($.%10 != 2))) {print $_}' | gzip -c > splits/train.gz
zcat labels.gz | perl -lane 'if ($. > 1 && ($.%10==1)) {print $_}' | gzip -c  > splits/valid.gz
zcat labels.gz | perl -lane 'if ($. > 1 && ($.%10==2)) {print $_}' | gzip -c > splits/test.gz
