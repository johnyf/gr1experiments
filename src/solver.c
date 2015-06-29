/* An efficient GR(1) synthesizer */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <dddmp.h>
#include <cudd.h>

/* also: try with multiple managers,
 * in a sequential program,
 * to observe the effect of decoupling the variable order
 */

mgr;
winningPositions;
initSys;
initEnv;
preVars;
livenessGuarantees;
strategyDumpingData;
variables;
safetyEnv;
variableTypes;
realizable;
postVars;
varCubePre;
variableNames;
varVectorPre;
varVectorPost;
varCubePostOutput;
addVariable;
computeVariableInformation;

DdManager *mgr;
DdNode *node;
Cudd_RecursiveDeref(mgr, node)
Cudd_Ref(node)
Cudd_Not(Cudd_ReadOne(mgr))
Cudd_CountPathsToNonZero(node); /* number of minterms */
Cudd_IsConstant(node);
Cudd_DagSize(node);
Cudd_NodeReadIndex(node);
Cudd_bddComputeCube(mgr, vars2, const_cast<int*> (phase), n)

Cudd_bddNewVar(mgr)
Cudd_bddExistAbstract(mgr, node, cube.cube)
Cudd_bddExistAbstract(mgr, node, cube.cube)
Cudd_bddUnivAbstract(mgr, node, cube.cube)
Cudd_bddSwapVariables(mgr, node, x.nodes, y.nodes, x.nofNodes)

mgr = Cudd_Init(0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, (long) 16 * 1024UL * 1024UL * 1024UL);
Cudd_SetMaxCacheHard(mgr, (unsigned int) -1);
Cudd_AutodynEnable(mgr, CUDD_REORDER_GROUP_SIFT);
Cudd_SetMaxGrowth(mgr, 1.2f);
Cudd_SetMinHit(mgr, 1.0);
setAutomaticOptimisation(true);

Cudd_AutodynEnable(mgr, CUDD_REORDER_SAME);

int nofLeft = Cudd_CheckZeroRef(mgr);
if (nofLeft != 0) {
	std::cerr << "Warning: " << nofLeft << " referenced nodes in the BDD manager left on destruction!\n";
}
Cudd_Quit(mgr);

Cudd_bddLeq(mgr, node, other.node);
Cudd_bddLeq(mgr, other.node, node);
Cudd_bddLeq(mgr, node, other.node) && (node != other.node);

Cudd_PrintInfo(mgr, stdout);


inline BFBdd operator|=(const BFBdd& other) {
	DdNode *result = Cudd_bddOr(mgr, node, other.node);
	Cudd_Ref(result);
	Cudd_RecursiveDeref(mgr, node);
	node = result;
	return *this;
}



void main(){
	DdNode* result;

	parse_args();
	init();
	result = compute_winning_set();
	
}


void compute_winning_set(){
	struct timeval start_time, end_time, diff_time;
	double time;
	gettimeofday(&start_time, NULL);
	
	std::vector<BF> Yj;
	
	unsigned int i, j, k, n;
	
	BF primed_nu2;
	BF livetransitions;
	BF goodForAnyLivenessAssumption;
	BF foundPaths;
	BF temp;
	BF newCases;
	BF casesCovered;
	BF strategy;

	BFFixedPoint nu2(mgr.constantTrue());
	for (; !nu2.isFixedPointReached(); ){
		primed_nu2 = nu2.getValue().SwapVariables(varVectorPre, varVectorPost);
		
		for (j=0; j<livenessGuarantees.size(); j++){
			printf("Guarantee: %d\n", j);
			livetransitions = livenessGuarantees[j] & primed_nu2;
			
			BFFixedPoint mu1(mgr.constantFalse());
			for (; !mu1.isFixedPointReached(); ){
				printf("mu1 iteration\n");
				
				livetransitions |= mu1.getValue().SwapVariables(varVectorPre, varVectorPost);
				goodForAnyLivenessAssumption = mu1.getValue();
				
				for (i=0; i<livenessAssumptions.size(); i++){
					printf("Assumption: %d\n", i);
					
					BFFixedPoint nu0(mgr.constantTrue());		
					for (; !nu0.isFixedPointReached(); ){
						printf("nu0 iteration\n");
						
						temp = nu0.getValue().SwapVariables(varVectorPre, varVectorPost);
						temp &= ! livenessAssumptions[i];
						temp |= livetransitions;
						temp &= safetySys;
						printf("rho_s & (livetransitions | (! J_i^e & nu0')\n");
						
						/* cox */
						temp = temp.ExistAbstract(varCubePostOutput);
						printf("exists done\n");
						
						temp = safetyEnv.Implies(temp);
						printf("implies done\n");
						
						temp = temp.UnivAbstract(varCubePostInput);
						printf("forall done\n");
						
						nu0.update(temp);
						temp = mgr.constantTrue();
						
						/* print info */
						gettimeofday(&end_time, NULL);
						timersub(&end_time, &start_time, &diff_time);
						time = diff_time.tv_sec * 1000 + diff_time.tv_usec / 1000;
						printf("time (ms): %1.3f, reordering (ms): %ld, sysj: %d, envi: %d, nodes: all: %ld, Z: %d, Y: %d, X: %d\n",
							time, Cudd_ReadReorderingTime(mgr.mgr),
							j, i,
							Cudd_ReadNodeCount(mgr.mgr),
							nu2.dag_size(), mu1.dag_size(), nu0.dag_size());
					}
					printf("take union of nu0\n");
					goodForAnyLivenessAssumption |= nu0.getValue();
				}
				mu1.update(goodForAnyLivenessAssumption);
			}
			printf("push back\n");
			Yj.push_back(BF(&mgr, mu1.getValue().getCuddNode()));
		}
		
		if (Cudd_DebugCheck(mgr.mgr) == 1){
			assert(false);
		}
		
		// conjoin
		while (Yj.size() > 1){
			n = Yj.size();
			k = floor(n / 2);
			printf("n = %d, k = %d\n", n, k);
			// consume the power of 2
			for (j=0; j<k; j++){
				Yj[j] = Yj[2 * j] & Yj[2 * j + 1];
			}
			// is there a last element ?
			if (2 * k < n){
				Yj[k] = Yj[2 * k];
			}
			// empty the tail
			for (j=0; j<k; j++){
				Yj.pop_back();
			}
			printf("len(Yj) = %lu\n", Yj.size());
			assert(Yj.size() == n - k);
		}
		
		if (Cudd_DebugCheck(mgr.mgr) == 1){
			assert(false);
		}
		assert(Yj.size() == 1);
		nu2.update(Yj[0]);
		Yj.pop_back();
		assert(Yj.size() == 0);
		
		if (Cudd_DebugCheck(mgr.mgr) == 1){
			assert(false);
		}
		
		/*
		BF nextContraintsForGoals = mgr.constantTrue();
		for (j=0; j<livenessGuarantees.size(); j++){
			BF y = Yj.back();
			Yj.pop_back();
			nextContraintsForGoals &= y;
		}
		nu2.update(nextContraintsForGoals);
		*/
	}
	winningPositions = nu2.getValue();
	
	printf("fixpoint found\n\n\n---------------------------\n");

	// start from the found fixpoint, just to fill the `strategyDumpingData`
	primed_nu2 = nu2.getValue().SwapVariables(varVectorPre, varVectorPost);
	strategyDumpingData.clear();
	for (unsigned int j=0; j<livenessGuarantees.size(); j++){
		livetransitions = livenessGuarantees[j] & primed_nu2;
		
		strategy = mgr.constantFalse();
		casesCovered = mgr.constantFalse();
		
		BFFixedPoint mu1(mgr.constantFalse());
		for (; !mu1.isFixedPointReached(); ) {
			livetransitions |= mu1.getValue().SwapVariables(varVectorPre, varVectorPost);
			goodForAnyLivenessAssumption = mu1.getValue();
			
			for (unsigned int i=0; i<livenessAssumptions.size(); i++){
				BFFixedPoint nu0(mgr.constantTrue());
				
				for (; !nu0.isFixedPointReached(); ){
					temp = nu0.getValue().SwapVariables(varVectorPre, varVectorPost);
					temp &= ! livenessAssumptions[i];
					temp |= livetransitions;
					temp &= safetySys;
					
					foundPaths = temp;
					
					temp = temp.ExistAbstract(varCubePostOutput);
					temp = safetyEnv.Implies(temp);
					temp = temp.UnivAbstract(varCubePostInput);
					
					nu0.update(temp);
					temp = mgr.constantTrue();
					
					/* print info */
					gettimeofday(&end_time, NULL);
					timersub(&end_time, &start_time, &diff_time);
					time = diff_time.tv_sec * 1000 + diff_time.tv_usec / 1000;
					printf("time (ms): %1.3f, reordering (ms): %ld, sysj: %d, envi: %d, nodes: all: %ld, Z: %d, Y: %d, X: %d\n",
						time, Cudd_ReadReorderingTime(mgr.mgr),
						j, i,
						Cudd_ReadNodeCount(mgr.mgr),
						nu2.dag_size(), mu1.dag_size(), nu0.dag_size());
				}
				goodForAnyLivenessAssumption |= nu0.getValue();
				
				newCases = foundPaths.ExistAbstract(varCubePostOutput) & !casesCovered;
				strategy |= newCases & foundPaths;
				casesCovered |= newCases;
			}
			mu1.update(goodForAnyLivenessAssumption);
		}
		strategyDumpingData.push_back(std::pair<unsigned int,BF>(j, strategy));
	}
	
	gettimeofday(&end_time, NULL);
	timersub(&end_time, &start_time, &diff_time);
	time = diff_time.tv_sec * 1000 + diff_time.tv_usec / 1000;
	printf("time (ms): %1.3f, reordering (ms): %ld, nodes: all: %ld\n----------\n",
		time, Cudd_ReadReorderingTime(mgr.mgr),
		Cudd_ReadNodeCount(mgr.mgr));
}


void make_street_1_transducer(std::string filename){
	mgr.printStats();
	printf("starting strategy synthesis.\n");
	struct timeval start_time, end_time, diff_time;
	unsigned int i, j, k, n;
	int selector;
	std::vector<int> counterVarNumbers;
	double time;
	gettimeofday(&start_time, NULL);
	
	BF counter;
	std::vector<BF> strategies(livenessGuarantees.size());
	
	// Allocate counter variables
	for (i=1; i<=livenessGuarantees.size(); i = i << 1){
		std::ostringstream os;
		os << "_jx_b" << counterVarNumbers.size();
		counterVarNumbers.push_back(addVariable(SymbolicStrategyCounterVar, os.str()));
	}
	selector = addVariable(SymbolicStrategyCounterVar, "strat_type");
	computeVariableInformation();
	
	// Prepare positional strategies for the individual goals
	for (i=0; i<livenessGuarantees.size(); i++){
		counter = mgr.constantTrue();
		for (j=0; j<counterVarNumbers.size(); j++){
			if (i & (1 << j)){
				counter &= variables[counterVarNumbers[j]];
			} else {
				counter &= !variables[counterVarNumbers[j]];
			}
		}
		strategies[i] = strategyDumpingData[i].second & counter &
			((!variables[selector]) | livenessGuarantees[i]);
	}
	strategyDumpingData.clear();
	
	// disjoin
	while (strategies.size() > 1){
		n = strategies.size();
		k = floor(n / 2);
		// consume the power of 2
		for (j=0; j<k; j++){
			strategies[j] = strategies[2 * j] | strategies[2 * j + 1];
		}
		// is there a last element ?
		if (2 * k < n){
			strategies[k] = strategies[2 * k];
		}
		// empty the tail
		for (j=0; j<k; j++){
			strategies.pop_back();
		}
		assert(strategies.size() == n - k);
		/* print info */
		gettimeofday(&end_time, NULL);
		timersub(&end_time, &start_time, &diff_time);
		time = diff_time.tv_sec * 1000 + diff_time.tv_usec / 1000;
		printf("time (ms): %1.3f, reordering (ms): %ld, nodes: all: %ld\n",
			time, Cudd_ReadReorderingTime(mgr.mgr),
			Cudd_ReadNodeCount(mgr.mgr));
	}
	assert(strategies.size() == 1);
	
	/* dump file */
	/*
	std::ostringstream fileExtraHeader;
	fileExtraHeader << "# This file is a BDD exported by the SLUGS\n#\n# This BDD is a strategy.\n";
	fileExtraHeader << "#\n# This header contains extra information used by LTLMoP's BDDStrategy.\n";
	fileExtraHeader << "# Currently, the only metadata is 1) the total number of system goals\n";
	fileExtraHeader << "# and 2) the mapping between variable numbers and proposition names.\n#\n";
	fileExtraHeader << "# Some special variables are also added:\n";
	fileExtraHeader << "#       - `_jx_b*` are used as a binary vector (b0 is LSB) to indicate\n";
	fileExtraHeader << "#         the index of the currently-pursued goal.\n";
	fileExtraHeader << "#       - `strat_type` is a binary variable used to indicate whether we are\n";
	fileExtraHeader << "#          moving closer to the current goal (0) or transitioning to the next goal (1)\n#\n";
	fileExtraHeader << "# Num goals: " << livenessGuarantees.size() << "\n";
	fileExtraHeader << "# Variable names:\n";
	for (i=0; i<variables.size(); i++) {
		fileExtraHeader << "#\t" << i << ": " << variableNames[i] << "\n";
	}
	fileExtraHeader << "#\n# For information about the DDDMP format, please see:\n";
	fileExtraHeader << "#    http://www.cs.uleth.ca/~rice/cudd_docs/dddmp/dddmpAllFile.html#dddmpDump.c\n#\n";
	fileExtraHeader << "# For information about how this file is generated, please see the SLUGS source.\n#\n";
	
	mgr.writeBDDToFile(filename.c_str(), fileExtraHeader.str(),
		strategies[0], variables, variableNames);
	printf("done writing DDDMP file.\n");
	*/
	/* print info */
	gettimeofday(&end_time, NULL);
	timersub(&end_time, &start_time, &diff_time);
	time = diff_time.tv_sec * 1000 + diff_time.tv_usec / 1000;
	printf("time (ms): %1.3f, reordering (ms): %ld, nodes: %ld\n",
		time, Cudd_ReadReorderingTime(mgr.mgr), Cudd_ReadNodeCount(mgr.mgr));
}


/*
int storeReturnValue = Dddmp_cuddBddStore(
mgr,
NULL,
bdd.getCuddNode(),
(char**)varNamesChar, // char ** varnames, IN: array of variable names (or NULL)
idMatcher,
DDDMP_MODE_TEXT,
// DDDMP_VARNAMES,
DDDMP_VARIDS,
NULL,
file
);
*/

