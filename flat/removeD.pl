#!/bin/env perl

use strict;
use warnings;

while(<>){
    chomp;
    if($_ =~ m/^>(\S+)\t(.+)/){
	my $aa = $1;
	my $rest = $2;
	$rest =~ s/^D-//;
	print '>'.join("\t", $aa, $rest)."\n";
    }else{
	print "$_\n";
    }
}
