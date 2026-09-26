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
#include <math.h>


double max(double x,double y) {
  if (x>y) {
      return x;
  }else{
      return y;
  }
}

double sigmoid(double x) {
    return 1/(1+exp(-x));
}



class Matrix {
private:
    std::vector<std::vector<double>> matrix;

public:

    Matrix() {};

    Matrix(const std::vector<double>& v) {
        matrix.push_back(v);
    }


    double get(int pos1,int pos2) {
        return matrix.at(pos1).at(pos2);
    }

    void set(int row,int coln,double value) {
        matrix.at(row).at(coln)=value;
    }

    void multiply(const Matrix& m1,const Matrix& m2) {
        matrix.assign(m1.matrix.size(),std::vector<double>(m2.matrix[0].size(), 0.0)); //product result
        if (m1.matrix[0].size()!=m2.matrix.size()) {
            throw std::runtime_error("Invalid Multiplication");
        }else {
            for (int i=0;i<m1.matrix.size();i++) {
                for (int j=0;j<m2.matrix[0].size();j++) {
                    double sum =0;
                    for (int k=0;k<m1.matrix[0].size();k++) {
                        sum+=m1.matrix[i][k]*m2.matrix[k][j];
                    }
                    matrix[i][j]=sum;
                }
            }
        }

    }

    void add(const Matrix& m1,const Matrix& m2) {
        matrix.assign(m1.matrix.size(),std::vector<double>(m2.matrix[0].size(), 0.0));//added result

        if (m1.matrix[0].size()!=m2.matrix[0].size() || m1.matrix.size()!=m2.matrix.size()) {
            throw std::runtime_error("Invalid Addition");
        }

        for (int i=0;i<m1.matrix.size();i++) {
            for (int j=0;j<m1.matrix[0].size();j++) {
                double sum=0;
                sum+=m1.matrix[i][j]+m2.matrix[i][j];
                matrix[i][j]=sum;
            }
        }
    }

    void subtract(const Matrix& m1,const Matrix& m2) {
        matrix.assign(m1.matrix.size(),std::vector<double>(m2.matrix[0].size(), 0.0));

        if (m1.matrix[0].size()!=m2.matrix[0].size() || m1.matrix.size()!=m2.matrix.size()) {
            throw std::runtime_error("Invalid Addition");
        }

        for (int i=0;i<m1.matrix.size();i++) {
            for (int j=0;j<m1.matrix[0].size();j++) {
                double result=0;
                result+=m1.matrix[i][j]-m2.matrix[i][j];
                matrix[i][j]=result;
            }
        }
    }

    void transpose(const Matrix& m) {
        matrix.assign(m.matrix[0].size(),std::vector<double>(m.matrix.size(), 0.0));

        for (int i=0;i<m.matrix.size();i++) {
            for (int j=0;j<m.matrix[0].size();j++) {
                matrix[j][i]=m.matrix[i][j];
            }
        }
    }

    void HP(const Matrix& m1, const Matrix& m2) {
        if (m1.matrix.size() != m2.matrix.size() ||
            m1.matrix[0].size() != m2.matrix[0].size()) {
            throw std::runtime_error("Invalid Hadamard Product");
            }

        matrix.assign(
            m1.matrix.size(),
            std::vector<double>(m1.matrix[0].size(), 0.0)
        );

        for (int i = 0; i < m1.matrix.size(); i++) {
            for (int j = 0; j < m1.matrix[0].size(); j++) {
                matrix[i][j] = m1.matrix[i][j] * m2.matrix[i][j];
            }
        }
    }

    void SM(const Matrix& m, double factor) {
        matrix.assign(
            m.matrix.size(),
            std::vector<double>(m.matrix[0].size(), 0.0)
        );

        for (int i = 0; i < m.matrix.size(); i++) {
            for (int j = 0; j < m.matrix[0].size(); j++) {
                matrix[i][j] = m.matrix[i][j] * factor;
            }
        }
    }

    void RIC(double lower_cap, double higher_cap, int rows, int columns) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(lower_cap, higher_cap);

        matrix.assign(
            rows,
            std::vector<double>(columns, 0.0)
        );

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = dist(gen);
            }
        }
    }


    //this is for using Relu as a activation function
    void ReluActivation(const Matrix& m) {
        matrix.assign(m.matrix.size(),std::vector<double>(m.matrix[0].size(), 0.0));
        for (int i = 0; i < m.matrix.size(); i++) {
            for (int j = 0; j < m.matrix[0].size(); j++) {
                matrix[i][j] = max(0,m.matrix[i][j]);
            }
        }
    }

    void sigmoidActivation(const Matrix& m) {
        matrix.assign(m.matrix.size(),std::vector<double>(m.matrix[0].size(), 0.0));
        for (int i = 0; i < m.matrix.size(); i++) {
            for (int j = 0; j < m.matrix[0].size(); j++) {
                matrix[i][j] = sigmoid(m.matrix[i][j]);
            }
        }
    }

    //to calculate initial loss
    double loss(const Matrix& predicted, const Matrix& actual) {
        if (predicted.matrix.size() != actual.matrix.size() ||
            predicted.matrix[0].size() != actual.matrix[0].size()) {
            throw std::runtime_error("Invalid Loss Calculation");
        }

        double total_loss = 0.0;
        for (int i = 0; i < predicted.matrix.size(); i++) {
            for (int j = 0; j < predicted.matrix[0].size(); j++) {
                double diff = predicted.matrix[i][j] - actual.matrix[i][j];
                total_loss += diff * diff;
            }
        }
        return total_loss / (predicted.matrix.size() * predicted.matrix[0].size());
    }

    void Relu_Derivative(const Matrix& m) {
        matrix.assign(m.matrix.size(),std::vector<double>(m.matrix[0].size(), 0.0));
        for (int i = 0; i < m.matrix.size(); i++) {
            for (int j = 0; j < m.matrix[0].size(); j++) {
                if (m.matrix[i][j]>0) {
                    matrix[i][j]=1;
                }else {
                    matrix[i][j]=0;
                }
            }
        }
    }

    void print() {
        for (int i=0;i<matrix.size();i++) {
            for (int j=0;j<matrix[0].size();j++) {
                std::cout<<matrix[i][j]<<" ";
            }
            std::cout<<"\n";
        }
    }
            
};

class Layer{
public:
    Matrix weights; //
    Matrix bias;
    Matrix pre_act;

    bool use_activation;

    Layer(int input_size,int output_size,bool activation=true) {
        weights.RIC(-0.5,0.5,input_size,output_size);
        bias.RIC(-0.5,0.5,1,output_size);
        use_activation=activation;
    }

    Matrix Activation(Matrix input) {
        Matrix result;
        result.ReluActivation(input);
        return result;
    }


    Matrix forward(Matrix input) {
        Matrix res,result;
        res.multiply(input,weights);
        result.add(res,bias);
        pre_act=result;
        if (use_activation) {
            result=Activation(result);
        }
        return result;
    }

    Matrix backward(Matrix input, Matrix grad_output, double learning_rate) {
        Matrix grad_input,grad_weights,grad_bias, weights_T,dz,dy,input_t;
        if (use_activation) {
            dy.Relu_Derivative(pre_act);
            dz.HP(grad_output,dy);
        }else {
            dz=grad_output;
        }
        weights_T.transpose(weights);
        input_t.transpose(input);
        grad_input.multiply(dz,weights_T);
        grad_weights.multiply(input_t,dz);
        grad_bias=dz;


        //update weights and bias
        Matrix scaled_grad_weights,scaled_grad_bias;
        scaled_grad_weights.SM(grad_weights,-learning_rate);
        scaled_grad_bias.SM(grad_bias,-learning_rate);

        Matrix weights_c=weights;
        Matrix bias_c=bias;

        weights.add(weights_c,scaled_grad_weights);
        bias.add(bias_c,scaled_grad_bias);

        return grad_input;
    }



};

class Network {
public:

    std::vector<Layer> layers;

    Network() {
        layers.emplace_back(784, 128,true);
        layers.emplace_back(128, 64,true);
        layers.emplace_back(64, 10,false);
    }

    void train(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& data,int epochs,double learning_rate) {

        for (int e=0;e<epochs;e++) {
            double average_loss=0;
            for (const auto& sample_data:data) {
                const auto& labelv=sample_data.first;
                const auto& pixelv=sample_data.second;
                Matrix input(pixelv);
                Matrix output(labelv);
                Matrix o1=layers[0].forward(input);
                Matrix o2=layers[1].forward(o1);
                Matrix o3=layers[2].forward(o2);
                Matrix lossM,Grad_O,temp;
                double loss_value = lossM.loss(o3, output);
                Grad_O.subtract(o3,output);
                temp = Grad_O;
                Grad_O.SM(temp,0.2);

                Matrix b1=layers[2].backward(o2,Grad_O,learning_rate);
                Matrix b2=layers[1].backward(o1,b1,learning_rate);
                Matrix b3=layers[0].backward(input,b2,learning_rate);

                average_loss+=loss_value;

            }
            std::cout<<"One epoch Average Loss Value: "<<average_loss/data.size()<<std::endl;
        }
    }
};



std::vector<std::pair<std::vector<double>,std::vector<double>>> read_csv(std::string filename) {
    std::vector<std::pair<std::vector<double>,std::vector<double>>> result;

    std::ifstream myFile(filename);
    if (!myFile.is_open()) {
        throw std::runtime_error("Couldnt open the csv data file");
    }

    std::string line,colname;
    int val;

    while (std::getline(myFile,line)) {
        std::stringstream ss(line);
        std::pair<std::vector<double>, std::vector<double>> row;
        std::vector<double> label(10,0.0);
        ss>>val;
        label[val]=1;
        if (ss.peek()==',') ss.ignore();
        std::vector<double> pixels;
        while (ss>>val){
            pixels.push_back(static_cast<double>(val) / 255.0);
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
    std::vector<std::pair<std::vector<double>, std::vector<double>>> data=read_csv("../data/mnist_test_3.csv");
    // int sample = 0;
    //
    // for (const auto& [labelv, pixelv] : data) {
    //     std::cout << "Sample: " << sample << "\n";
    //     std::cout << "Label: " ;
    //
    //     for (const auto& label : labelv) {
    //         std::cout<<label<<" ";
    //     }
    //
    //     std::cout<<"\n";
    //
    //     std::cout << "Pixels: ";
    //
    //     for (const auto& pixel : pixelv) {
    //         std::cout << pixel << " ";
    //     }
    //
    //     std::cout << "\n\n";
    //     sample++;
    // }


    //for testing this matrix

    // Matrix m1,m2,m3;
    // m2.RIC(5,10,2,2);
    // m1.RIC(5,10,2,2);
    // m3.add(m1,m2);
    //
    // m1.print();
    // m2.print();
    // m3.print();


    Network n;

    n.train(data,10,0.001);


    return EXIT_SUCCESS;
}


