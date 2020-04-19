#include <iostream>
#include <opencv2/opencv.hpp>
//#include <highgui.h>
#include <stdio.h>
//#include <algorithm>
#include <math.h>
#include <vector>
#include<memory>
#include<time.h>


#include "MNN/ImageProcess.hpp"
#include "MNN/MNNForwardType.h"
#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/Matrix.h"
#include "MNN/Rect.h"

using namespace std;
using namespace cv;
using namespace MNN;
Mat ProcessImg(Mat img,int size)
{	
	
	
	Mat dst;
	float h = img.rows;
	float w = img.cols;
	float scale = min(size/w,size/h);
	int nw = (int)w*scale;
	int nh = (int)h*scale;
	resize(img,img,Size(nw,nh));

	Mat img2(size, size, img.type(), Scalar(128, 128, 128));
	//Mat imageROI = img2(Rect((size-nw)/2, (size-nh)/2, img.cols, img.rows)); 
	//img.copyTo(imageROI,img);
	
	//dst.create(img2.size(), img2.type());	
	int row = (size-nh)/2;
	int col =(size-nw)/2;
	while (row <= (img.rows+(size-nh)/2))
	{
		col = (size-nw)/2;
		while (col <= (img.cols+(size-nw)/2))
		{
			int b = img.at<Vec3b>(row-(size-nh)/2, col-(size-nw)/2)[0];
            int g = img.at<Vec3b>(row-(size-nh)/2, col-(size-nw)/2)[1];
            int r = img.at<Vec3b>(row-(size-nh)/2, col-(size-nw)/2)[2];
            img2.at<Vec3b>(row, col)[0] = b;
            img2.at<Vec3b>(row, col)[1] = g;
            img2.at<Vec3b>(row, col)[2] = r;
			col++;
		}
		row ++;
	}
	//imwrite("image2.jpg",img2);
	//imwrite("image.jpg",img);

	/*
	for(int row = (size-nh)/2; row < img2.rows; row++)
    {
        for(int col =(size-nw)/2; col < img2.cols; col++)
        {
            int b = img2.at<Vec3b>(row, col)[0];
            int g = img2.at<Vec3b>(row, col)[1];
            int r = img2.at<Vec3b>(row, col)[2];
            dst.at<Vec3b>(row, col)[0] = b/255.;
            dst.at<Vec3b>(row, col)[1] = g/255.;
            dst.at<Vec3b>(row, col)[2] = r/255.;

        }
    }*/
	//img2 = cv::transpose(img2,[2,0,1]);
	
	return img2;
}

int main(){
	char const *filename = "/home/zsharp/sdk_develop/rubbish/data/image.jpg";
	//conv2d_10/BiasAdd conv2d_13/BiasAdd Inputs: input_1
	string model_name = "/home/zsharp/sdk_develop/rubbish/data/tiny/tiny.mnn";
	int precision = 2;
    int power     = 0;
    int memory    = 0;
    int threads   = 1;
	int INPUT_SIZE = 416;
	int forward = MNN_FORWARD_CPU;
	// load image && peocessimg
	
	Mat image;
	image = imread(filename);
	int input_shape[2] = {image.cols,image.rows};
	
	image = ProcessImg(image,INPUT_SIZE);

	// load and config mnn model
	auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_name.c_str()));
	
	if (nullptr == net) {
		cout << "load model failed." << endl;
		return 10000;
	}
	/*
    auto revertor = unique_ptr<Revert>(new Revert(model_name.c_str()));
    revertor->initialize();
    auto modelBuffer      = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    auto net = shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    revertor.reset();
	*/
    MNN::ScheduleConfig config;
    config.numThread = threads;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = (MNN::BackendConfig::PowerMode) power;
    backendConfig.memory = (MNN::BackendConfig::MemoryMode) memory;
    config.backendConfig = &backendConfig;

    auto session = net->createSession(config);
	net->releaseModel();
	
	if (nullptr == session) {
		cout << "[LOG]create session failed." << endl;
		return 10000;
	}

    //net->releaseModel();
    

	clock_t start = clock();
    // preprocessing

    image.convertTo(image, CV_32FC3);
    //image = (image * 2 / 255.0f) - 1;
	//imwrite("demo.jpg",image);
    // wrapping input tensor, convert nhwc to nchw    
	std::vector<int> dims{1,3, INPUT_SIZE, INPUT_SIZE};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);

    std::string input_tensor = "input_1";
    auto inputTensor  = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);


	// run network
	
	net->runSession(session);
	
     // get output data

	 
    string output_tensor_name0 = "conv2d_10/BiasAdd";
    string output_tensor_name1 = "conv2d_13/BiasAdd";


    auto tensor_boxes  = net->getSessionOutput(session, output_tensor_name0.c_str());
    auto tensor_cls   = net->getSessionOutput(session, output_tensor_name1.c_str());

    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
    MNN::Tensor tensor_cls_host(tensor_cls, tensor_cls->getDimensionType());
    

    tensor_boxes->copyToHostTensor(&tensor_boxes_host);
    tensor_cls->copyToHostTensor(&tensor_cls_host);
	
	clock_t end = clock();
	double times = (double)(end-start)/CLOCKS_PER_SEC;
	cout <<"[LOG]Times: "<< times <<endl;
	
    auto boxes_dataPtr   = tensor_boxes_host.host<float>();
    auto cls_dataPtr = tensor_cls_host.host<float>();
	
	
	cout << boxes_dataPtr << endl;
	cout << cls_dataPtr << endl;
	
	return 0;
}
//g++ test.cpp -o test `pkg-config --libs --cflags opencv,mnn` -ldl
