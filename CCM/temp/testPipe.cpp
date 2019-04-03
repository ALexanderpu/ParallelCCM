/**
 * @author Bo Pu
 * @email pubo.alexander@gmail.com
 * @create date 2018-08-27 11:01:41
 * @modify date 2018-08-27 11:01:41
 * @desc [description]
*/

#include <bits/stdc++.h>
#include "include/json.hpp"

using namespace std;
using json = nlohmann::json;
// run using command
//cd "/Users/alexpb/Desktop/Lab/PySparkCUDAC/" && g++ testPipe.cpp -std=c++11 -o testPipe  &&./testPipe
int main(int argc, char *argv[]){
    // para parse
    if(argc > 1){
        for(auto arg = 1; arg < argc; arg++){
            cout << argv[arg] <<endl;
        }
        return 0;
    }
    int l, tau, e, num_samples;
    vector<float> observation, target;
    cout << argv[1] << endl;
    json j = json::parse(argv[1]);
    for(auto& ele:j["observation"]){
        observation.emplace_back(ele);
    }
    for(auto& ele:j["target"]){
        target.emplace_back(ele);
    }
    l = j["l"];
    tau = j["tau"];
    num_samples = j["samples"];
    e = j["e"];
    /*
    cout << tau << " " << e << " " << l << " " << num_samples <<endl;
    for(auto ele: observation){
        cout << ele << " ";
    }
    cout << endl;
    for(auto ele:target){
        cout << ele << " ";
    }
    cout << endl;
    */
   cout << l << tau << e;
    //json j_vec(c_vector);

    return 0;
}
