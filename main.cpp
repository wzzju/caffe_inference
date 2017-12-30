#include <iostream>
#include <fstream>
#include "cnpy.h"
#include "utilities.h"
#include "helper.h"
#include "net.h"

using namespace std;
using namespace chrono;

void test_net_single() {
    /****************************构建网络****************************/
    string proto_path = "/sdspace/deeplearning/Caffe_Code_Analysis/caffe_extractor/lenet/";
    net m_net(proto_path);
    /****************************准备输入数据****************************/
    vector<float> input(INPUT_C * INPUT_H * INPUT_W);
    string img_path = "/sdspace/mnist_images/test/4/00033.png";
    get_image_data(img_path, input.data());
//    cnpy::npy_save("mydata.npy", input_data, {3, 28, 28}, "w");
    /****************************前向推断****************************/
    vector<float> result = m_net.forward(input.data());
    float max = 0;
    int max_i = -1;
    for (int i = 0; i < 10; ++i) {
        float value = result[i];
        if (max < value) {
            max = value;
            max_i = i;
        }
    }
    cout << "Max prob : " << max << ",  Class : " << max_i << endl;
}

//CORRECT: 9916
//TOTAL: 10000
//ACCURACY: 0.9916
//Cost 100.654 seconds
void test_net_acc() {
    /****************************构建网络****************************/
    string proto_path = "/sdspace/deeplearning/Caffe_Code_Analysis/caffe_extractor/lenet/";
    net m_net(proto_path);

    vector<float> input(INPUT_C * INPUT_H * INPUT_W);
    vector<float> result;
    ifstream in("mnist/test/test.txt");
    char buffer[256];
    stringstream stream;
    string file_name;
    int label;
    unsigned long total = 0, correct = 0;
    while (!in.eof()) {
        in.getline(buffer, 100);
        stream << buffer;
        stream >> file_name;
        stream >> label;
        stream.str("");
        stream.clear();
        /****************************准备输入数据****************************/
        get_image_data(file_name, input.data());
//    cnpy::npy_save("mydata.npy", input_data, {3, 28, 28}, "w");
        /****************************前向推断****************************/
        result = m_net.forward(input.data());
        float max = 0;
        int max_i = -1;
        for (int i = 0; i < 10; ++i) {
            float value = result[i];
            if (max < value) {
                max = value;
                max_i = i;
            }
        }
        if (max_i == label)
            correct++;
        total++;
    }
    cout << "CORRECT : " << correct << endl;
    cout << "TOTAL : " << total << endl;
    cout << "ACCURACY : " << (double) correct / (double) total << endl;
}

int main() {
//    auto start = system_clock::now();
//    test_net_acc();
//    auto end = system_clock::now();
//    auto duration = duration_cast<microseconds>(end - start);
//    cerr << "Cost " << static_cast<double>(duration.count()) * microseconds::period::num / microseconds::period::den
//         << " seconds" << endl;
    {
        CostTimeHelper timeHelper("test_net_single");
        test_net_single();
    }
    {
        CostTimeHelper timeHelper("test_net_acc");
        test_net_acc();
    }
    return 0;
}