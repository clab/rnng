#!/usr/bin/perl -w
use strict;
while(<>) {
  chomp;
  my @toks = split /\s+/;
  my $sentid = shift @toks;
  shift @toks;
  my $score = shift @toks;
  shift @toks;
  print "$sentid ||| $score |||";
  for my $t (@toks) {
    if ($t =~ /^\(/) {
      print " $t";
    } else {
      print " (XX $t)";
    }
  }
  print "\n";
}

