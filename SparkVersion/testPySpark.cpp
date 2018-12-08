/**
 * @author Bo Pu
 * @email pubo.alexander@gmail.com
 * @create date 2018-08-27 07:25:14
 * @modify date 2018-08-27 07:25:14
 * @desc [description]
*/
#include "include/json.hpp"
#include <bits/stdc++.h>

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
        vector<double> observation, target;
        json j;
        try{
            j = json::parse(str);
        }catch(json::parse_error){
            cout << "error: parse input json string" << endl;
            return 0;
        }
        for(auto &ele: j["observation"]){
            observation.emplace_back(ele);
        }
        for(auto &ele:j["target"]){
            target.emplace_back(ele);
        }
        e = (size_t)j["e"];
        lib_size = (size_t)j["l"];
        tau = (size_t)j["tau"];
        samples = (size_t)j["samples"];
        // for testing input
        // cout << "ttreceived, e: " << e << " tau:" << tau << " l: " << lib_size << " samples:" << samples << endl;
        // run ccm algorithm
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
        vector<double> result = {1, 2, 3, 4, 5, 6};
        json jresult;
        jresult["result"] = result;
        string resultJson = jresult.dump();
        cout << resultJson << endl;
    }
    return 0;
}