#!/usr/bin/env perl -w 

###############################################
# perl script to generated LTL specification and
# config file to synthesize a generalized buffer
# between k senders and 2 receivers.
#
# Usage: ./genbuf_generator.pl <num_of_senders> <fname>
#
# Generated files: fname
# Default file name is genbuf
#
###############################################
use strict;
use POSIX; # qw(ceil floor);

sub slc {
    my $j = shift;
    my $bits = shift;
    my $assign = "";
    my $val;

    for (my $bit = 0; $bit < $bits; $bit++) {
	$val = $j % 2;
	if ($val == 0){
		$assign .= "!SLC$bit'";
	}else{
		$assign .= "SLC$bit'";
	}
	$assign .= " & " unless ($bit == $bits-1);
	$j = floor($j/2);
    }
    return $assign;
}

###############################################
# MAIN

if(! defined($ARGV[0])) {
    print "Usage: ./genbuf_generator.pl <num_of_senders> <prefix>\n";
    exit;
}

my $fname = "genbuf";
if( defined($ARGV[1])) {
    $fname = $ARGV[1];
}

#variables for LTL specification
my $num_senders = $ARGV[0];
my $slc_bits = ceil((log $num_senders)/(log 2));
$slc_bits = 1 if ($slc_bits == 0);
my $num_receivers = 2;
my $guarantees = "";
my @assert;
my $assumptions = "";
my @assume;

#variables for config file
my $input_vars  = "";
my $output_vars = "";
my $env_initial = "";
my $sys_initial = "";
my $env_transitions = "";
my $sys_transitions = "";
my $env_fairness = "";
my $sys_fairness = "";

###############################################
# Communication with senders
#
my ($g, $a);
for (my $i=0; $i < $num_senders; $i++) {
    #variable definition
    $input_vars .= "StoB_REQ$i\n";
    $output_vars.= "BtoS_ACK$i\n";

    #initial state
    $env_initial .= "! StoB_REQ$i\n";
    $sys_initial .= "! BtoS_ACK$i\n";
    
    $guarantees .= "\n##########################################\n";
    $guarantees .= "#Guarantees for sender $i\n";
    
    ##########################################
    # Guarantee 1
    $g = "StoB_REQ$i -> F(BtoS_ACK$i)  \n # G1\n";
    $guarantees .= $g; push (@assert, $g);
    # Guarantee 2
    $g = "(! StoB_REQ$i & StoB_REQ$i') -> (! BtoS_ACK$i')  \n # G2\n";
    $guarantees .= $g; push (@assert, $g);
    $sys_transitions .= $g;
    $sys_fairness .= "StoB_REQ$i <-> BtoS_ACK$i  \n # G1 + G2\n";
    
    # Guarantee 3
#    $g = "G(StoB_REQ$i=0 -> (BtoS_ACK$i=1 + X(BtoS_ACK$i=0)));\t#G3\n";
    $g = "(! BtoS_ACK$i & ! StoB_REQ$i) -> (! BtoS_ACK$i')  \n # G2\n";
    $guarantees .= $g; push (@assert, $g);
    $sys_transitions .= $g;
    
    # Guarantee 4
    $g = "(BtoS_ACK$i & StoB_REQ$i) -> (BtoS_ACK$i')  \n # G4\n";
    $guarantees .= $g; push (@assert, $g);
    $sys_transitions .= $g;
    
    # Assumption 1
    $a = "(StoB_REQ$i & ! BtoS_ACK$i) -> (StoB_REQ$i')  \n # A1\n";
    $assumptions .= $a; push (@assume, $a);
    $env_transitions .= $a;
    $a = "BtoS_ACK$i -> (! StoB_REQ$i')  \n # A1\n";
    $assumptions .= $a; push (@assume, $a);
    $env_transitions .= $a;
    
    # Guarantee 5
    for (my $j=$i+1; $j < $num_senders; $j++) {
	$g = "(! BtoS_ACK$i) | (! BtoS_ACK$j)  \n # G5\n";
	$guarantees .= $g; push (@assert, $g);
	$sys_transitions .= $g;
    }
}

###############################################
# Communication with receivers
#
if ($num_receivers != 2) {
    print "Note that the DBW for Guarantee 7 works only for two receivers.\n";
    exit;
}
for (my $j=0; $j < $num_receivers; $j++) {
    #variable definition
    $input_vars .= "RtoB_ACK$j\n";
    $output_vars.= "BtoR_REQ$j\n";

    #initial state
    $env_initial .= "! RtoB_ACK$j\n";
    $sys_initial .= "! BtoR_REQ$j\n";
    
    $guarantees .= "\n##########################################\n";
    $guarantees .= "# Guarantees for receiver $j\n";
    
    # Assumption 2
    $a = "G(BtoR_REQ$j -> F(RtoB_ACK$j))  \n # A2\n";
    $assumptions .= $a; push (@assume, $a);
    $env_fairness .= "BtoR_REQ$j <-> RtoB_ACK$j  \n # A2\n";
    
    # Assumption 3
    $a = "(! BtoR_REQ$j) -> (! RtoB_ACK$j')  \n # A3\n";
    $assumptions .= $a; push (@assume, $a);
    $env_transitions .= $a;
    
    # Assumption 4
    $a = "(BtoR_REQ$j & RtoB_ACK$j) -> (RtoB_ACK$j')  \n # A4\n";
    $assumptions .= $a; push (@assume, $a);
    $env_transitions .= $a;

    # Guarantee 6
    $g = "(BtoR_REQ$j & ! RtoB_ACK$j) -> (BtoR_REQ$j')  \n # G6\n";
    $guarantees .= $g; push (@assert, $g);
    $sys_transitions .= $g;

    # Guarantee 7
    for (my $k=$j+1; $k < $num_receivers; $k++) {
	$g = "(! BtoR_REQ$j) | (! BtoR_REQ$k)  \n # G7\n";
	$guarantees .= $g; push (@assert, $g);
	$sys_transitions .= $g;
    }
    # G7: rose($j) -> X (no_rose W rose($j+1 mod $num_receivers))
    my $n = ($j + 1)%$num_receivers; #next
    my $rose_j  = "( (! BtoR_REQ$j) & (BtoR_REQ$j'))";
    my $nrose_j = "( BtoR_REQ$j | (! BtoR_REQ$j'))";
    my $rose_n  = "( (! BtoR_REQ$n) & (BtoR_REQ$n'))";
    $g = "G( $rose_j ->\n X(($nrose_j U $rose_n) + \n G($nrose_j)))  \n # G7\n";
    $guarantees .= $g; push (@assert, $g);
    #construct DBW for G7 - see below
    
    # Guarantee 6 and 8
    $g = "RtoB_ACK$j -> (! BtoR_REQ$j')  \n # G8\n";
    $guarantees .= $g; push (@assert, $g);
    $sys_transitions .= $g;
}

# DBW for guarantee 7
$output_vars .= "stateG7_0\n";
$output_vars .= "stateG7_1\n";
$sys_initial .= "! stateG7_0\n";
$sys_initial .= "stateG7_1\n";
$sys_transitions .= "(BtoR_REQ0 & BtoR_REQ1) -> FALSE  \n # G7\n";
$sys_transitions .= "( (! stateG7_1) & (! BtoR_REQ0) & BtoR_REQ1) ->";
$sys_transitions .= " (stateG7_1' & (! stateG7_0') )  \n # G7\n";
$sys_transitions .= "(stateG7_1 & BtoR_REQ0 & (! BtoR_REQ1) ) ->";
$sys_transitions .= " ( (! stateG7_1') & (! stateG7_0') )  \n # G7\n";
$sys_transitions .= "( (! stateG7_1) & (! BtoR_REQ0) & (! BtoR_REQ1) ) ->";
$sys_transitions .= " ( (! stateG7_1') & stateG7_0')  \n # G7\n";
$sys_transitions .= "(stateG7_1 & (! BtoR_REQ0) & (! BtoR_REQ1) ) ->";
$sys_transitions .= " (stateG7_1' & stateG7_0')  \n # G7\n";
$sys_transitions .= "( (! stateG7_1) & (! stateG7_0) & BtoR_REQ0 & (! BtoR_REQ1) ) ->";
$sys_transitions .= " ( (! stateG7_1') & (! stateG7_0') )  \n # G7\n";
$sys_transitions .= "(stateG7_1 & (! stateG7_0) & (! BtoR_REQ0) & BtoR_REQ1) ->";
$sys_transitions .= " (stateG7_1' & (! stateG7_0') )  \n # G7\n";
$sys_transitions .= "( (! stateG7_1) & stateG7_0 & BtoR_REQ0) -> FALSE  \n # G7\n";
$sys_transitions .= "(stateG7_1 & stateG7_0 & BtoR_REQ1) -> FALSE  \n # G7\n";

###############################################
# Communication with FIFO and multiplexer
#
#variable definition
$input_vars .= "FULL\n";
$input_vars .= "EMPTY\n";
$output_vars .= "ENQ\n";
$output_vars .= "DEQ\n";
$output_vars .= "stateG12\n";

#initial state
$env_initial .= "! FULL\n";
$env_initial .= "EMPTY\n";
$sys_initial .= "! ENQ\n";
$sys_initial .= "! DEQ\n";
$sys_initial .= "! stateG12\n";

for (my $bit=0; $bit < $slc_bits; $bit++) {
    $output_vars .= "SLC$bit\n";
    $sys_initial .= "! SLC$bit\n";
}

$guarantees .= "\n##########################################\n";
$guarantees .= "# Guarantees for FIFO and multiplexer\n";

# Guarantee 9: ENQ and SLC
$guarantees .= "\n##########################################\n";
$guarantees .= "# ENQ <-> Exists i: rose(BtoS_ACKi)\n";
my $roseBtoS  = "";
my $roseBtoSi = "";
for (my $i=0; $i < $num_senders; $i++) {
    $roseBtoSi = "( (! BtoS_ACK$i) & BtoS_ACK$i')";
    $g = "$roseBtoSi -> ENQ'  \n # G9\n";
    $guarantees .= $g; push (@assert, $g);
    $sys_transitions .= $g;
    
    $roseBtoS   .=   "(BtoS_ACK$i | (! BtoS_ACK$i'))";
    $roseBtoS   .= " &   " if ($i < ($num_senders - 1));
    if ($i == 0) {
	$g = "$roseBtoSi  -> (".slc($i, $slc_bits).")  \n # G9\n";
    } else {
	$g = "$roseBtoSi <-> (".slc($i, $slc_bits).")  \n # G9\n";
    }
    $guarantees .= $g; push (@assert, $g);
    $sys_transitions .= $g;
}
$g = "($roseBtoS) -> ! ENQ'  \n # G9\n";
$guarantees .= $g; push (@assert, $g);
$sys_transitions .= $g;

# Guarantee 10
$guarantees .= "\n##########################################\n";
$guarantees .= "# DEQ <-> Exists j: fell(RtoB_ACKj)\n";
my $fellRtoB = "";
for (my $j=0; $j < $num_receivers; $j++) {
    $g = "(RtoB_ACK$j & (! RtoB_ACK$j')) -> DEQ'  \n # G10\n";
    $guarantees .= $g; push (@assert, $g);
    $sys_transitions .= $g;
    $fellRtoB   .=   "( (! RtoB_ACK$j) | RtoB_ACK$j')";
    $fellRtoB   .= " &   " if ($j < ($num_receivers - 1));
}
$g = "($fellRtoB) -> (! DEQ')  \n # G10\n";
$guarantees .= $g; push (@assert, $g);
$sys_transitions .= $g;

# Guarantee 11
$guarantees .= "\n";
$g = "(FULL & (! DEQ)) -> (! ENQ)  \n # G11\n";
$guarantees .= $g; push (@assert, $g);
$sys_transitions .= $g;

$g = "EMPTY -> (! DEQ)  \n # G11\n";
$guarantees .= $g; push (@assert, $g);
$sys_transitions .= $g;

# Guarantee 12
$g = "G( (! EMPTY) -> F(DEQ))  \n # G12\n";
$guarantees .= $g; push (@assert, $g);
$sys_transitions .= "( (! stateG12) & EMPTY) -> (! stateG12')  \n # G12\n";
$sys_transitions .= "( (! stateG12) & DEQ  ) -> (! stateG12')  \n # G12\n";
$sys_transitions .= "( (! stateG12) & (! EMPTY) & (! DEQ) ) -> stateG12'  \n # G12\n";
$sys_transitions .= "(stateG12 & (! DEQ) ) -> stateG12'  \n # G12\n";
$sys_transitions .= "(stateG12 & DEQ  ) -> (! stateG12')  \n # G12\n";
$sys_fairness .= "! stateG12  \n # G12\n";

# Assumption 4
$a = "(ENQ & (! DEQ) ) -> ! EMPTY'  \n # A4\n";
$assumptions .= $a; push (@assume, $a);
$env_transitions .= $a;

$a = "(DEQ & (! ENQ) ) -> ! FULL'  \n # A4\n";
$assumptions .= $a; push (@assume, $a);
$env_transitions .= $a;

$a  = "(ENQ <-> DEQ) -> ((FULL <-> FULL') & (EMPTY <-> EMPTY'))  \n # A4\n";
$assumptions .= $a; push (@assume, $a);
$env_transitions .= $a;
  
###############################################
# PRINT CONFIG FILE
###############################################
print "Generating $fname\n";
open (CFG, ">$fname");

print CFG "###############################################\n";
print CFG "# Input variable definition\n";
print CFG "###############################################\n";
print CFG "[INPUT]\n";
print CFG $input_vars;
print CFG "\n";

print CFG "###############################################\n";
print CFG "# Output variable definition\n";
print CFG "###############################################\n";
print CFG "[OUTPUT]\n";
print CFG $output_vars;
print CFG "\n";

print CFG "###############################################\n";
print CFG "# Environment specification\n";
print CFG "###############################################\n";
print CFG "[ENV_INIT]\n";
print CFG $env_initial;
print CFG "\n";
print CFG "[ENV_TRANS]\n";
print CFG $env_transitions;
print CFG "\n";
print CFG "[ENV_LIVENESS]\n";
print CFG $env_fairness;
print CFG "\n";

print CFG "###############################################\n";
print CFG "# System specification\n";
print CFG "###############################################\n";
print CFG "[SYS_INIT]\n";
print CFG $sys_initial;
print CFG "\n";
print CFG "[SYS_TRANS]\n";
print CFG $sys_transitions;
print CFG "\n";
print CFG "[SYS_LIVENESS]\n";
print CFG $sys_fairness;
print CFG "\n";
close CFG;

