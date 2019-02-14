/**
 * @author Bo Pu
 * @email pubo.alexander@gmail.com
 * @create date 2018-08-27 07:26:45
 * @modify date 2018-08-27 07:26:45
 * @desc class to parse config file (.ini) at the beginning of CCM Parallel Algorithm
*/
#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "global.h"

class ConfigReader
{
	private:
		// data sturcture to store contents in config file (section, name, value)
		std::unordered_map<std::string, std::unordered_map<std::string, std::string> > records;

		// check only if ' '; '\t'; '\n'; '\f'; '\r' exist
		// usually it is to split words in the sentence
		bool space_only(const std::string& line){
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
			auto last_non_whitespace = prev(std::find_if_not(s.rbegin(), s.rend(), is_whitespace).base());
			s.erase(next(last_non_whitespace), end(s));
			return s;
		}
	
		// erase the middle space
		std::string& normalize(std::string& s) {
		    s.erase(std::remove_if(begin(s), end(s),[] (char c) { return c == ' ' || c == '\t'; }), end(s));
		    return s;
		}
	
		// validate the line: if it is ilegal, skip; else we take the settings and store in records
		bool is_valid (std::string & line){
			line = trim(line);
		    line = normalize(line);
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
	
	// init with file (call rea_file to parse)
        explicit ConfigReader (const std::string & file){  read_file (file); }
	
	// read and parsing at the same time
        bool read_file (const std::string& file);
	
	// query config settings
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


#endif
