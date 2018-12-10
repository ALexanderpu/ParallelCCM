#include "global.h"

// parse time-series  two columns: the first one is x and the second one is y
std::pair<std::vector<float>, std::vector<float> > parse_csv(std::string &csvfile){
    std::ifstream data(csvfile);
    std::string line;
    std::vector<std::vector<float> > csvdata;
    unsigned long length = 0;
    while(getline(data, line)){
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> parsedRow;
        while(getline(lineStream,cell,',')) //include head
        {
            parsedRow.push_back(strtof(cell.c_str(), 0));
        }
        length += 1;
        csvdata.push_back(parsedRow);
    }
    std::vector<float> x, y;
    std::cout << length-1 << " size "<< std::endl;
    for(int i = 1; i < length; i++){
        x.push_back(csvdata[i][1]);
        y.push_back(csvdata[i][2]);
    }
    return std::make_pair(x, y);
}

// write down the result to testify & compare plot accuracy
void dump_csv(std::string &csvfile, std::vector<float>& rhos, size_t E, size_t tau, size_t lib_size){
	std::ofstream resultfile;
	resultfile.open(csvfile);
	std::string header = "E, tau, L, rho\n";
	resultfile << header;
    for(size_t r = 0; r < rhos.size(); r++){
        resultfile << E << ", " << tau << ", " << lib_size << ", " << rhos[r] << std::endl;
    }
	resultfile.close();
}