#include "mpi.h"
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
class ConfigReader
{
	private:
		// section, name, value
		std::unordered_map<string, unordered_map<string, string>> records;

		// check only if ' '; '\t'; '\n'; '\f'; '\r' exist
		// usually it is to split words in the sentence
		bool space_only(const string& line){
			for(size_t i = 0; i < line.length();i++){
				if(std::isspace((unsigned char)line[i]))
					return false;
			}
			return true;
		}
		// erase the leading space and trailing space
		std::string& trim(std::string& s){
		    auto is_whitespace = [] (char c) -> bool { return c == ' ' || c == '\t'; };
			auto first_non_whitespace = std::find_if_not(s.begin(), s.end(), is_whitespace);
			s.erase(begin(s), first_non_whitespace);
			auto last_non_whitespace = std::find_if_not(s.rbegin(), s.rend(), is_whitespace).base();
			s.erase(next(last_non_whitespace), end(s));
			return s;
		}
		// erase the middle space
		std::string& normalize(std::string& s) {
		    s.erase(std::remove_if(begin(s), end(s),[] (char c) { return c == ' ' || c == '\t'; }), end(s));
		    return s;
		}

		bool is_valid (std::string & line){
			trim(line);
		    normalize (line);
		    std::size_t i = 0;
		    // if the line is a section
		    if (line[i] == '[')
		    {
		        // find where the section's name ends
		        std::size_t j = line.find_last_of (']');
		        // if the ']' character wasn't found, then the line is invalid.
		        if (j == std::string::npos)
		            return false;
		        // if the distance between '[' and ']' is equal to one,
		        // then there are no characters between section brackets -> invalid line.
		        if (j - i == 1)
		            return false;
		    }
		    /* Check if a line is a comment */
		    else if (line[i] == ';' || line[i] == '#' || (line[i] == '/' && line[i + 1] == '/'))
		        return false;
		    /* Check if a line is ill-formed */
		    else if (line[i] == '=' || line[i] == ']')
		        return false;
		    else // is key=value pattern?
		    {
		        std::size_t j = line.find_last_of ('=');
		        if (j == std::string::npos)
		            return false;
		        if (j + 1 >= line.length())
		            return false;
		    }
		    return true;
		}
		void parse(std::string& section, std::string& line){
			std::size_t i = 0;
			if(line[i] == '['){
				i++;
				std::size_t j = line.find_last_of(']')-1;
				section = line.substr(i, j);
			}else{
				std::string sec(section);
				std::size_t j = line.find ('=');
				std::string name = line.substr (i, j);
				std::string value = line.substr(j+1);
				records[sec][name] = value;
			}
		}
	public:

        ConfigReader(){};

        explicit ConfigReader (const std::string & file){
        	read_file (file);
        }

        bool read_file (const std::string& file);

        std::string get_string (const std::string & tsection, const std::string & tname);

};

bool ConfigReader::read_file(const std::string& file){
	records.clear();
	std::ifstream config(file);
	if(!config.is_open())
		return false;
	std::string section;
	std::string buffer;

	while(std::getline(config, buffer, '\n')){
		if(is_valid(buffer)){
			parse(section, buffer);
		}else{
			return false;
		}
	}
	return true;
}

std::string ConfigReader::get_string (const std::string & tsection, const std::string & tname){
	return records[tsection][tname];
}


struct IncGenerator {
    int current_, interval_;
    IncGenerator (int start, int interval) : current_(start),interval_(interval) {}
    int operator() () { return current_ += interval_; }
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

// write down the result to testify & compare plot accuracy
void dump_csv(std::string &csvfile, size_t E, size_t tau, unordered_map<size_t, vector<float>>& rho_bins){
	std::ofstream resultfile;
	resultfile.open(csvfile);
	std::string header = "E, tau, L, rho\n";
	resultfile << header;
	for(auto it = rho_bins.begin(); it != rho_bins.end(); it++){
		for(size_t r = 0; r < it->second.size(); r++){
			resultfile << E << ", " << tau << ", " << it->first << ", " << it->second[r] << endl;
		}
	}
	resultfile.close();
}

// ccm function only execute for a part of parameter sets and input data
// OpenMP and Thrust used
std::vector<float> ccm(vector<float> observations, vector<float> targets, size_t E, size_t tau, size_t lib_size, int num_samples){
    // return rhos
    vector<float > rhos = {1,2,3,5,7};
    omp_set_num_threads(2);
    #pragma omp parallel for
    for(size_t i = 0; i < rhos.size(); i++){
        rhos[i] += E;
    }
    return rhos;
}

int main(int argc, char **argv){
    string file = "/home/bo/cloud/CCM-Parralization/ccm.cfg";
    ConfigReader cr;
    int num_samples;
    std::vector<int> EArr, tauArr, LArr;
    string input, output;
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
    cout << "Process ID " << my_id << " is on " << processor_name << endl;

    if(my_id == 0){
        // read settings from local file
        
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
            num_vectors = observations.size();
        }catch(int e){
            cout << "loading data and parameter error" <<endl;
            return 1;
        }

        if(observations.size() != targets.size()){
            cout << "input sequence length not match" << endl;
            return 1;
        }
        /*
        num_vectors = 5;
        
        observations.assign(num_vectors, 3);
        targets.assign(num_vectors, 5);
        */
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

    MPI_Bcast(&num_vectors, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&elements_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);


    if(my_id != 0){
        observations.resize(num_vectors); 
        targets.resize(num_vectors);
        combinationEArr.resize(num_tasks);
        combinationtauArr.resize(num_tasks);
        combinationLArr.resize(num_tasks);
    }

    MPI_Bcast(&observations[0], observations.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&targets[0], targets.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(&combinationEArr[0], combinationEArr.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&combinationtauArr[0], combinationtauArr.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&combinationLArr[0], combinationLArr.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // call ccm here
    int start_index = my_id*elements_per_proc;
    int end_index = start_index + elements_per_proc-1;

    for(int i = start_index; i <= end_index; i++){
        
        vector<float> result = ccm(observations, targets, combinationEArr[i], combinationtauArr[i], combinationLArr[i], num_samples);
        cout << "task " << i << " has the first result: " << result[0] << endl;
        // better to save result into csv here
    }
    
    
    // finally gather the result
    MPI_Finalize();
    return 0;
}