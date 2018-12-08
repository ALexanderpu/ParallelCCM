#include "ccm.h"
#include "include/json.hpp"
using namespace std;
using json = nlohmann::json;
pair<vector<double>, vector<double> > parse_csv(std::string &csvfile){
    std::ifstream data(csvfile);
    std::string line;
    std::vector<std::vector<double> > csvdata;
    unsigned long length = 0;
    while(getline(data, line)){
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<double> parsedRow;
        while(getline(lineStream,cell,',')) //include head
        {
            parsedRow.push_back(strtof(cell.c_str(), 0));
        }
        length += 1;
        csvdata.push_back(parsedRow);
    }
    vector<double> x, y;
    for(int i = 1; i < length; i++){
        x.push_back(csvdata[i][2]);
        y.push_back(csvdata[i][3]);
    }
    return make_pair(x, y);
}

int main(int argc, char *argv[]){



    // input should be replaced
    string csvfile = "../data/two_species_model.csv";
    pair<vector<double>, vector<double> > input = parse_csv(csvfile);
    size_t e = 2;
    size_t tau = 1;
    auto lib_size = size_t(0.2*input.first.size());
    size_t samples = 300;

    CCM model = CCM(false, false);
    model.init(input.first, input.second, e, tau, lib_size, samples);
    model.run();
    vector<double> result = model.get_prediction();
    json j;
    j["result"] = result;
    std::string resultJson = j.dump();
    cout << resultJson;
    return 0;
}