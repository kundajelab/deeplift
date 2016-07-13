perl -le 'foreach my $i(1..700) {print "negSynth$i\nposSynth$i"}' > train.txt
perl -le 'foreach my $i(701..850) {print "negSynth$i\nposSynth$i"}' > valid.txt
perl -le 'foreach my $i(851..1000) {print "negSynth$i\nposSynth$i"}' > test.txt
