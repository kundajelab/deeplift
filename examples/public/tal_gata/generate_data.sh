#!/usr/bin/env bash

##simdata generation - install simdna
densityMotifSimulation.py --seed 1234 --prefix gata --motifNames GATA_disc1 --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200 --numSeqs 1000
densityMotifSimulation.py --seed 1234 --prefix tal --motifNames TAL1_known1 --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200 --numSeqs 1000
densityMotifSimulation.py --seed 1234 --prefix talgata --motifNames GATA_disc1 TAL1_known1 --max-motifs 3 --min-motifs 1 --mean-motifs 2 --seqLength 200 --numSeqs 1000

rm *_info.txt
rm *.fa
gzip -f *.simdata
