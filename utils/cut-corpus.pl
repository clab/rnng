#!/usr/bin/perl -w
use strict;
die "Usage: $0 N\nSplits a corpus separated by ||| symbols and returns the Nth field\n" unless scalar @ARGV > 0;

my $x = shift @ARGV;
my @ind = split /,/, $x;
my @o = ();
for my $ff (@ind) {
  if ($ff =~ /^\d+$/) {
    push @o, $ff - 1;
  } elsif ($ff =~ /^(\d+)-(\d+)$/) {
    my $a = $1;
    my $b = $2;
    die "$a-$b is a bad range in input: $x\n" unless $b > $a;
    for (my $i=$a; $i <= $b; $i++) {
      push @o, $i - 1;
    }
  } else {
    die "Bad input: $x\n";
  }
}

while(<>) {
  chomp;
  my @fields = split /\s*\|\|\|\s*/;
  my @sf;
  for my $i (@o) {
    my $y = $fields[$i];
    if (!defined $y) { $y= ''; }
    push @sf, $y;
  }
  print join(' ||| ', @sf) . "\n";
}


