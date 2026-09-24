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


std::vector<std::pair<int,std::vector<int>>> read_csv(std::string filename) {
    std::vector<std::pair<int,std::vector<int>>> result;

    std::ifstream myFile(filename);
    if (!myFile.is_open()) {
        throw std::runtime_error("Couldnt open the csv data file");
    }

    std::string line,colname;
    int val;

    while (std::getline(myFile,line)) {
        std::stringstream ss(line);
        std::pair<int, std::vector<int>> row;
        int label;
        ss>>label;
        if (ss.peek()==',') ss.ignore();
        std::vector<int> pixels;
        while (ss>>val){
            pixels.push_back(val);
            if (ss.peek()==',') ss.ignore();
        }
        row.first=label;
        row.second=pixels;

        result.push_back(row);

    }
    myFile.close();
    return result;

}


int main() {
    std::vector<std::pair<int, std::vector<int>>> data=read_csv("../data/mnist_train.csv");
    int sample = 0;

    for (const auto& [digit, pixelv] : data) {
        std::cout << "Sample: " << sample << "\n";
        std::cout << "Label: " << digit << "\n";
        std::cout << "Pixels: ";

        for (const auto& pixel : pixelv) {
            std::cout << pixel << " ";
        }

        std::cout << "\n\n";
        sample++;
    }
    return EXIT_SUCCESS;
}