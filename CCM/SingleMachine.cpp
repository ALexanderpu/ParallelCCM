#include "include/csv.hpp"
#include "include/ccm.hpp"
#include "include/config.hpp"
using namespace std;
int main(int argc, char *argv[])
{
    // test accuracy and function of ccm
    string cfgfile = "/home/bo/cloud/CCM-Parralization/ccm.cfg";
    ConfigReader cr;
    cr.read_file(cfgfile);

    
    string input = cr.get_string("paths", "input");
    string output = cr.get_string("paths", "output");

    vector<float > observations;
    vector<float > targets;
    std::tie(observations, targets) = parse_csv(input);
    if(observations.size() != targets.size()){
    	cout << "input sequence length not match" << endl;
    	return 1;
    }

    size_t num_vectors = observations.size();
    int num_samples = 250;
    size_t E = 3;
    size_t tau = 1;
    size_t lib_size = min((size_t)700, num_vectors);
    vector<float> result = ccm(observations, targets, E, tau, lib_size, num_samples);
    string output_file = "/E_" + to_string((int)E) + "_tau_" + to_string((int)tau) + "_l_" + to_string((int)lib_size) + "_samples_" + to_string(num_samples) + ".csv";
    string output_path = output + output_file;
    dump_csv(output_path, result, E, tau, lib_size);
}
