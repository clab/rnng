#!/usr/bin/perl -w
use strict;

die "Usage: $0 oracle.txt hyp-trees.txt\n" unless scalar @ARGV == 2;

open O, "<$ARGV[0]" or die "Can't read $ARGV[0]: $!";
open H, "<$ARGV[1]" or die "Can't read $ARGV[1]: $!";

while(<O>) {
  next unless /^# \(/;
  <O>;
  my $sent = <O>;
  <O>;
  chomp $sent;
  my @toks = split /\s+/, $sent;
  my $hyptree = <H>;
  die "fewer sentences in hypothesis than in reference" unless defined $hyptree;
  chomp $hyptree;
  $hyptree =~ s/\) \)/))/g;
  $hyptree =~ s/\) \)/))/g;
  my @htoks = split /\s+/, $hyptree;
  shift @htoks;
  shift @htoks;
  shift @htoks;
  shift @htoks;
  my @otoks = ();
  my $i = 0;
  for my $ht (@htoks) {
    if ($ht =~ /^([^)]+)(\)+)$/) {
      my $t = $toks[$i];
      die unless defined $t;
      $i++;
      push @otoks, "$t$2";
    } else {
      push @otoks, $ht;
    }
  }
  print "@otoks\n";
}

