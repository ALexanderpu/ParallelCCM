#include "mpi.h"
#include "include/ccm.hpp"
#include "include/config.hpp"
#include "include/csv.hpp"

using namespace std;

int main(int argc, char **argv){
    
    if(argc <= 1){
        std::cout << "please pass the config file as the first argument!" << std::endl;
        return 1;
    }
    ConfigReader cr;
    int num_samples;
    std::vector<int> EArr, tauArr, LArr;
    std::string input, output;
    vector<float > observations, targets;
    int num_vectors;

    // count the time
    double start_time, end_time;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];


    std::vector<int> combinationEArr, combinationtauArr, combinationLArr;
    int elements_per_proc;

    int my_id, num_procs, num_tasks;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI::Get_processor_name(processor_name, namelen);
    std::cout << "Worker ID " << my_id << " is on " << processor_name << endl;
    bool flag = false;
    // read settings from local file
    string file(argv[1]);
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

        output = cr.get_string("paths", "output");

        if(my_id == 0){
            // read csv file for observations and targets to broadcast to other processor
            input = cr.get_string("paths", "input");
            string xname = cr.get_string("inputs", "x");
            string yname = cr.get_string("inputs", "y");
            std::tie(observations, targets) = parse_csv(input, xname, yname);
            num_vectors = observations.size();
            if(observations.size() != targets.size()){
                std::cout << "input sequence length not match" << std::endl;
                flag = true;
            }
            // broadcast tasks
            num_tasks = tauArr.size()*EArr.size()*LArr.size();
            elements_per_proc = num_tasks / num_procs; // should be dividable 

            for(auto e: EArr){
                for(int ele = 0; ele < tauArr.size()*LArr.size(); ele++){
                    combinationEArr.push_back(ele);
                }
                
                for(auto tau: tauArr){
                    for(int ele = 0; ele < LArr.size(); ele++){
                        combinationtauArr.push_back(ele);
                    }   
                    for(auto l: LArr){
                        combinationLArr.push_back(l);
                    }
                }
            }
        }

        MPI_Bcast(&flag, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    }catch(int e){
        std::cout << "loading data and parameter error" << std::endl;
        return 1;
    }

    if(flag){ // input sequence is not match
        return 1;
    }


    /*
    if(my_id == 0){
        // read settings from local file
        string file(argv[1]);
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
            input = cr.get_string("paths", "input");
            output = cr.get_string("paths", "output");
            cout << "output directory: " << output << endl;
            string xname = cr.get_string("inputs", "x");
            string yname = cr.get_string("inputs", "y");

            std::tie(observations, targets) = parse_csv(input, xname, yname);
            num_vectors = observations.size();
        }catch(int e){
            cout << "loading data and parameter error" <<endl;
            return 1;
        }

       // how to scatter the parameter settings, E tau, L to assign task to each processor?
       num_tasks = tauArr.size()*EArr.size()*LArr.size();
       elements_per_proc = num_tasks / num_procs; // should be dividable 

       for(auto e: EArr){
           for(int ele = 0; ele < tauArr.size()*LArr.size(); ele++){
                combinationEArr.push_back(ele);
           }
           
           for(auto tau: tauArr){
                for(int ele = 0; ele < LArr.size(); ele++){
                   combinationtauArr.push_back(ele);
                }   
                for(auto l: LArr){
                   combinationLArr.push_back(l);
                }
           }
       }
    }
    */

    MPI_Bcast(&num_vectors, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&elements_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);

     // initial the result array
    float *finalResult[num_tasks];
    for(int i = 0; i < num_tasks; i++){
        finalResult[i] = (float *) malloc(num_samples * sizeof(float));
    }



    if(my_id != 0){
        observations.resize(num_vectors); 
        targets.resize(num_vectors);
        combinationEArr.resize(num_tasks);
        combinationtauArr.resize(num_tasks);
        combinationLArr.resize(num_tasks);
    }

    MPI_Bcast(&observations[0], observations.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&targets[0], targets.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(&combinationEArr[0], combinationEArr.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&combinationtauArr[0], combinationtauArr.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&combinationLArr[0], combinationLArr.size(), MPI_INT, 0, MPI_COMM_WORLD);

    // call ccm here
    int start_index = my_id*elements_per_proc;
    int end_index = start_index + elements_per_proc - 1;

    for(int i = start_index; i <= end_index; i++){
        CCMParallel cp;
        vector<float> result = cp.ccm(observations, targets, combinationEArr[i], combinationtauArr[i], combinationLArr[i], num_samples);
        std::cout << "task " << i << " with id:  " << my_id << " has the first result: " << result[0] << " size is: " << result.size() << endl;
        // better to save result into csv here
        std::string csvfile = output +  "/e_" + to_string(combinationEArr[i]) + "_tau_" + to_string(combinationtauArr[i]) + "_l_" + to_string(combinationLArr[i]) + "_MPIversion.csv";
        std::cout << "task " << i << " with id:  " << my_id << "write file: " << csvfile <<endl;
        dump_csv(csvfile, result, (size_t)combinationEArr[i], (size_t)combinationtauArr[i], (size_t)combinationLArr[i]);
        //float* a = &result[0];
        //MPI_Gather(a, num_samples, MPI_FLOAT, finalResult[i], num_samples, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    /*
    if(my_id == 0){
        for(int i = 0; i < num_tasks; i++){
            cout << "task " << i << " has the first result: " << finalResult[i][0] << endl;
        }
    }
    */
    /*
    if(my_id == 0){
        // with the combinationE/tau/LArr  and finalResult
        size_t prev_E = combinationEArr[0];
        size_t prev_tau = combinationtauArr[0];
        std::string csvfile = output +  "/e_" + to_string(prev_E) + "_tau_" + to_string(prev_tau) + "_MPIversion.csv";
        std::unordered_map<size_t, std::vector<float> > rho_bins;
        int l_sizes = 0;
        for(int i = 0; i < num_tasks; i++){
            size_t cur_E = combinationEArr[i];
            size_t cur_tau = combinationtauArr[i];
            size_t cur_L = combinationLArr[i];
            
            if(cur_E == prev_E && cur_tau == prev_tau){
                // add finalResult into rho_bins;
                vector<float> rho(finalResult[i], finalResult[i] + num_samples);
                rho_bins[cur_L] = rho;
                l_sizes++;

            }else{
                dump_csv_multiLs(csvfile, rho_bins, prev_E, prev_tau);
                // start a new
                prev_E = cur_E;
                prev_tau = cur_tau;
                csvfile = output +  "/e_" + to_string(prev_E) + "_tau_" + to_string(prev_tau) + "_MPIversion.csv";
                l_sizes = 1;
                // clear rho_bins then add new
                rho_bins.clear();
                vector<float> rho(finalResult[i], finalResult[i] + num_samples);
                rho_bins[cur_L] = rho;
            }
        }
        if(l_sizes){
            dump_csv_multiLs(csvfile, rho_bins, prev_E, prev_tau);
        }
        
    }*/
    
    // free up the 2 dimensions array: finalResult
    /*
    for(int i = 0; i < num_tasks; ++i) {
        delete[] finalResult[i];   
    }
    //Free the array of pointers
    delete[] finalResult;
    */
    
    // finally gather the result
    MPI_Finalize();
    return 0;
}