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
#include <random>

class Matrix {
public:
    std::vector<std::vector<double>> matrix;

    double get(int pos1,int pos2,std::vector<std::vector<double>>& m) {

        double target;
        target=m.at(pos1).at(pos2);
        return target;
    }

    void set(int row,int coln,double value,std::vector<std::vector<double>>& m) {
        double& p=m.at(row).at(coln);
        p=value;
    }

    std::vector<std::vector<double>> multiply(std::vector<std::vector<double>> m1,std::vector<std::vector<double>> m2) {
        std::vector<std::vector<double>> m3(m1.size(),std::vector<double>(m2[0].size(), 0.0)); //product result
        if (m1[0].size()!=m2.size()) {
            throw std::runtime_error("Invalid Multiplication");
        }else {
            for (int i=0;i<m1.size();i++) {
                for (int j=0;j<m2[0].size();j++) {
                    double sum =0;
                    for (int k=0;k<m1[0].size();k++) {
                        sum+=m1[i][k]*m2[k][j];
                    }
                    m3[i][j]=sum;
                }
            }
        }
        return m3;
    }

    std::vector<std::vector<double>> add(std::vector<std::vector<double>> m1,std::vector<std::vector<double>> m2) {
        std::vector<std::vector<double>> m3(m1.size(),std::vector<double>(m2[0].size(), 0.0));//added result

        if (m1[0].size()!=m2[0].size() || m1.size()!=m2.size()) {
            throw std::runtime_error("Invalid Addition");
        }

        for (int i=0;i<m1.size();i++) {
            for (int j=0;j<m1[0].size();j++) {
                double sum=0;
                sum+=m1[i][j]+m2[i][j];
                m3[i][j]=sum;
            }
        }
        return m3;
    }

    std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> m) {
        std::vector<std::vector<double>> mt(m[0].size(),std::vector<double>(m.size(), 0.0));

        for (int i=0;i<m.size();i++) {
            for (int j=0;j<m[0].size();j++) {
                mt[j][i]=m[i][j];
            }
        }
        return mt;
    }

     std::vector<std::vector<double>> HP(std::vector<std::vector<double>> m1,std::vector<std::vector<double>> m2) {
        std::vector<std::vector<double>> m3(m1.size(),std::vector<double>(m1[0].size(), 0.0));
        if (m1[0].size()!=m2[0].size() || m1.size()!=m2.size()) {
            throw std::runtime_error("Invalid Addition");
        }

        for (int i=0;i<m1.size();i++) {
            for (int j=0;j<m2[0].size();j++) {
                double product=1;
                product*=m1[i][j]*m2[i][j];
                m3[i][j]=product;
            }
        }
        return m3;
    }

    std::vector<std::vector<double>>SM(std::vector<std::vector<double>> m,double factor) {
        std::vector<std::vector<double>> mp(m.size(),std::vector<double>(m[0].size(), 0.0));

        for (int i=0;i<m.size();i++) {
            for (int j=0;j<m[0].size();j++) {
                mp[i][j]=m[i][j]*factor;
            }
        }
        return mp;
    }

    std::vector<std::vector<double>>RIC(double lower_cap,double higher_cap,int rows,int columns) {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_real_distribution<double> dist(lower_cap, higher_cap);

        std::vector<std::vector<double>> m(rows,std::vector<double>(columns, 0.0));

        for (int i=0;i<m.size();i++) {
            for (int j=0;j<m[0].size();j++) {
                m[i][j]=dist(gen);
            }
        }

        return m;
    }


};


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


