#ifndef CSV_HPP
#define CSV_HPP

#include "global.h"
#include <fstream>
// parse time-series  two columns: the first one is x and the second one is y
std::pair<std::vector<float>, std::vector<float> > parse_csv(std::string &csvfile, std::string &xname, std::string &yname){
    std::ifstream data(csvfile);
    std::string line;
    std::vector<std::vector<float> > csvdata;
    unsigned long length = 0;
    // identify which col to parse
    bool head = true;
    int xcol = -1, ycol = -1;
    while(getline(data, line)){
        std::stringstream lineStream(line);
        std::string cell;
        if(head){
            int index = -1;
            while(getline(lineStream, cell, ',')){
                index++;
                if(cell == xname){
                    xcol = index;
                }else if(cell == yname){
                    ycol = index;
                }
            }
            head = false;
        }else{
            std::vector<float> parsedRow;
            while(getline(lineStream, cell, ',')){ 
                parsedRow.push_back(strtof(cell.c_str(), 0));
            }
            length++;
            csvdata.push_back(parsedRow);
        }
    }
    std::vector<float> x, y;
    std::cout << length << " size "<< std::endl;

    if(xcol != -1 && ycol != -1){
        for(int i = 0; i < length; i++){
            x.push_back(csvdata[i][xcol]);
            y.push_back(csvdata[i][ycol]);
        }
    }else{
        std::cout << "parsing error: xName and yName not found" << std::endl;
    }
    return std::make_pair(x, y);
}


void dump_csv(std::string csvfile, std::vector<float>& rhos, size_t E, size_t tau, size_t lib_size){
    std::ofstream resultfile;
	resultfile.open(csvfile);
    if(resultfile.is_open()){
        std::string header = "E, tau, L, rho\n";
        resultfile << header;
        for(size_t r = 0; r < rhos.size(); r++){
            resultfile << E << ", " << tau << ", " << lib_size << ", " << rhos[r] << std::endl;
        }
        resultfile.flush();
        resultfile.close();
    }else{
        std::cout << "can not create the file" << std::endl;
    }
    return;
}

// write down the result to testify & compare plot accuracy
void dump_csv_multiLs(std::string csvfile, std::unordered_map<size_t, std::vector<float> >& rho_bins, size_t E, size_t tau){
    std::ofstream resultfile;
	resultfile.open(csvfile);
    if(resultfile.is_open()){
        // std::cout << realpath(csvfile.c_str(), NULL) << std::endl;
        std::string header = "E, tau, L, rho\n";
        resultfile << header;
        for(auto it = rho_bins.begin(); it != rho_bins.end(); it++){
            for(size_t r = 0; r < it->second.size(); r++){
                resultfile << E << ", " << tau << ", " << it->first << ", " << it->second[r] << std::endl;
            }
        }
        resultfile.flush();
        resultfile.close();
    }else{
        std::cout << "can not create the file" << std::endl;
    }
    return;
}

#endif