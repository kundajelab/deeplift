#!/usr/bin/env bash
mkdir splits
zcat outputs.gz | perl -lane 'if ($. > 1 && (($.%10 != 1) && ($.%10 != 2))) {print $F[0]}' | gzip -c > splits/train.gz
zcat outputs.gz | perl -lane 'if ($. > 1 && ($.%10==1)) {print $F[0]}' | gzip -c  > splits/valid.gz
zcat outputs.gz | perl -lane 'if ($. > 1 && ($.%10==2)) {print $F[0]}' | gzip -c > splits/test.gz
