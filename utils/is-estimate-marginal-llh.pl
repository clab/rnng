#!/usr/bin/perl -w

my $num_sents = shift @ARGV;
my $num_samples_per_sent = shift @ARGV;

open Q, "<$ARGV[0]" or die;
open B, "<$ARGV[1]" or die;

my $ll = 0;
for (my $s = 0; $s < $num_sents; $s++) {
  my $linenum = $s + 1;
  my $tiw = 0;
  my $ltiw = undef;  # define undef to be -inf UGLY
  my $tlen = 0;
  my $c = 0;
  my $max = -9e99;
  my $max_tree;
  for (my $i = 0; $i < $num_samples_per_sent; $i++) {
    my $a = <Q>;
    chomp $a;
    my ($sentid, $q, $tree) = split /\s*\|\|\|\s*/, $a;
    die unless $sentid == $s;
    my $b = <B>;
    chomp $b;
    my ($len, $pxy) = split /\s+/, $b;
    #  next if $i > 100;
    if (-$pxy > $max) { $max = -$pxy; $max_tree = $tree; }
    die unless $q <= 0;
    die unless $pxy > 0;
    $tlen += $len;
    $c++;
    $pxy *= -1;
    $iw = $pxy - $q;
#    $tiw += exp($iw);
    $ltiw = logsumexp($ltiw, $iw);
    print "sent=$linenum sample=$i ||| proposal=$q ||| joint=$pxy\n";
  }
  $tiw /= $c;
  $tlen /= $c;
  $ll += $tlen;
  #$tiw = log($tiw);
  $ltiw -= log($c);
  $tt += $ltiw;
  die "Estimated log marginal likelihood is $ltiw" unless $ltiw < 0;
  print "prob: $ltiw (len=$tlen)\n";
  print STDERR "$s ||| $max ||| $max_tree\n";
}
print "len=$ll words\n";
print "LLH=$tt\n";
my $ppl = exp(-$tt/$ll);
print "PPL=$ppl\n";

sub logsumexp {
  my ($a,$b) = @_;
  if (!defined $a) { return $b; }
  if (!defined $b) { return $a; }
  if ($a < $b) { my $c = $a; $a = $b; $b= $c; }
  return $b + log(1 + exp($a - $b));
#  log1p( v_ = v_ + log1p(std::exp(a.v_ - v_));  
}
