void make_street_1_transducer(std::string filename){
	std::vector<int> counterVarNumbers;
	std::vector<BF> strategies(livenessGuarantees.size());
	// Allocate counter variables
	for (i=1; i<=livenessGuarantees.size(); i = i << 1){
		std::ostringstream os;
		os << "_jx_b" << counterVarNumbers.size();
		counterVarNumbers.push_back(
            addVariable(SymbolicStrategyCounterVar, os.str()));
	}
	selector = addVariable(SymbolicStrategyCounterVar, "strat_type");
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
}
