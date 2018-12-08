#include "mpi.h"
#include <bits/stdc++.h>
using namespace std;

class ConfigReader{
    
};

pair<vector<float>, vector<float> > parse_csv(std::string &csvfile){
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
    vector<float> x, y;
    cout << length-1 << " size "<< endl;
    for(int i = 1; i < length; i++){
        x.push_back(csvdata[i][1]);
        y.push_back(csvdata[i][2]);
    }
    return make_pair(x, y);
}

int main(int argc, char **argv){
    // read csv file broadcast to other processor
    // read parameters and scatter to other processor

    int n, myid, num_procs, i;
    // parse config file: /home/bo/cloud/CCM-Parralization/ccm.cfg

    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // finally gather the result
    return 0;
}