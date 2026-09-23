//
// Created by samar on 9/22/2026.
//

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
int main() {
    std::cout<<"Yo";
    return EXIT_SUCCESS;
}


void read_csv(std::string filename) {
    std::vector<std::pair<std::vector<int>,std::vector<int>>> result;

    std::ifstream myFile(filename);
    if (!myFile.is_open()) {
        throw std::runtime_error("Couldnt open the csv data file");
    }

    std::string line,column;
    int val;


    if (myFile.good()) {
        std::getline(myFile,line);
        std::stringstream ss(line);
        
    }


}
