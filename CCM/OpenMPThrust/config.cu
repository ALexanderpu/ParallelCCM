#include "csv.hpp"
#include "ccm.hpp"
#include "config.hpp"
#include <string>
using namespace std;
int main(int argc, char *argv[])
{
    // test accuracy and function of ccm
    string file = "/home/bo/cloud/CCM-Parralization/ccm.cfg";
    ConfigReader cr;
    int num_samples;
    std::vector<int> EArr, tauArr, LArr;
    string input, output;
    vector<float > observations, targets;
    try{
        // parse config file: /home/bo/cloud/CCM-Parralization/ccm.cfg
        cr.read_file(file);

        // read parameters and scatter to other processor
        string E = cr.get_string("parameters", "E");
        string tau = cr.get_string("parameters", "tau");

        // parse to int vector for parameter: tau, E
        std::stringstream Ess(E), tauss(tau);
        int i;
        while(Ess >> i){
            EArr.push_back(i);
            if(Ess.peek() == ',')
                Ess.ignore();
        }
        while(tauss >> i){
            tauArr.push_back(i);
            if(tauss.peek() == ',')
                tauss.ignore();
        }

        num_samples = stoi(cr.get_string("parameters", "num_samples"));

        // parse to int for parameter: L
        int LStart = stoi(cr.get_string("parameters", "LStart"));
        int LEnd = stoi(cr.get_string("parameters", "LEnd"));
        int LInterval = stoi(cr.get_string("parameters", "LInterval"));
        size_t Lsize = (LEnd-LStart)/LInterval + 1;

        LArr.assign(Lsize, 0);
        IncGenerator g (LStart-LInterval, LInterval);
        std::generate(LArr.begin(), LArr.end(), g);

        // read csv file for observations and targets to broadcast to other processor
        string input = cr.get_string("paths", "input");
        string output = cr.get_string("paths", "output");

        std::tie(observations, targets) = parse_csv(input);
        size_t num_vectors = observations.size();
        size_t e = EArr[0];
        size_t t = tauArr[0];
        cout << "inputs: e " << e << " t " << t << " num_samples: " << num_samples << endl;
        if(e > 0 && t > 0 && num_samples > 0 && LStart > 0 && LEnd > 0 && LInterval > 0 ){
            unordered_map<size_t, vector<float> > rho_bins;
            CCMParallel cp;
            for(size_t lib_size = LStart; lib_size <= LEnd; lib_size+=LInterval){
                vector<float> result = cp.ccm(observations, targets, e, t, lib_size, num_samples);
                rho_bins[lib_size] = result;
            }
            string output_file = "/singlemachine.csv";
            string output_path = output + output_file;
            dump_csv_multiLs(output_path, rho_bins, e, t);
        }else{
            cout << "the parameters of ccm is not correct" << endl;
            return 1;
        }
    }catch(int e){
        cout << "loading data and parameter error" <<endl;
        return 1;
    }
    if(observations.size() != targets.size()){
        cout << "input sequence length not match" << endl;
        return 1;
    }
    return 0;
}
