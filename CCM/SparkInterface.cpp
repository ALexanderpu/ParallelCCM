/**
 * @author Bo Pu
 * @email pubo.alexander@gmail.com
 * @create date 2018-08-27 07:25:14
 * @modify date 2018-08-27 07:25:14
 * @desc spark c script interface
*/
#include "include/json.hpp"
#include "include/ccm.hpp"
#include "include/csv.hpp"

using json = nlohmann::json;

using namespace std;

int main(int argc, char *argv[]){
    string str;
    while(1){
        if(!getline(cin, str)){
            return 0;
        }
        // parse str to get the parameters and input time_series
        size_t lib_size, tau, e, samples;
        vector<float> observations, targets;
        json j;
        try{
            j = json::parse(str);
        }catch(json::parse_error){
            cout << "error: parse input json string" << endl;
            return 0;
        }
        for(auto &ele: j["observations"]){
            observations.emplace_back(ele);
        }
        for(auto &ele:j["targets"]){
            targets.emplace_back(ele);
        }
        // how to differentiate multils
        bool is_multil = (size_t) j["multil"];
        e = (size_t)j["e"];
        
        tau = (size_t)j["tau"];
        samples = (size_t)j["samples"];
        if(is_multil == 0){
            lib_size = (size_t)j["l"];
            CCMParallel cp;
            vector<float> result = cp.ccm(observations, targets, e, tau, lib_size, samples);
            json jresult;
            jresult["e"] = e;
            jresult["tau"] = tau;
            jresult["l"] = lib_size;
            jresult["result"] = result;

            string resultJson = jresult.dump();
            cout << resultJson << endl;

        }else{
            size_t LStart = (size_t)j["lstart"];
            size_t LEnd = (size_t)j["lend"];
            size_t LInterval = (size_t)j["linterval"];
            unordered_map<size_t, vector<float> > rho_bins;
            CCMParallel cp;
            for(size_t lib_size = LStart; lib_size <= LEnd; lib_size+=LInterval){
                vector<float> result = cp.ccm(observations, targets, e, tau, lib_size, samples);
                rho_bins[lib_size] = result;
            }
            // generate csv file and cout the file name back  
            // later we need to check if c++ can write hdfs file system
            
            string output_file = "/e_" + to_string(e) + "_tau_" + to_string(tau) + "_sparkc.csv";
            string output_dir = "/home/bo/cloud/CCM-Parralization/Result";
            string output_path = output_dir + output_file;
            
            dump_csv_multiLs(output_path, rho_bins, e, tau);
            
            cout << output_path << endl;
        }
        // for testing input
        // cout << "ttreceived, e: " << e << " tau:" << tau << " l: " << lib_size << " samples:" << samples << endl;
        
        // run ccm algorithm
        
        // return all values
        
        
        // return average
        /*
        double average = 0;
        if(result.size() > 0){
            double total = accumulate(result.begin(), result.end(), 0.0);
            average = total / (1.0*result.size())
        }
        cout << average << endl;
        */
        // return whole result
    }
    return 0;
}