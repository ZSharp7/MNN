#include <iostream>
#include <opencv2/opencv.hpp>
//#include <highgui.h>
#include <stdio.h>
//#include <algorithm>
#include <math.h>
#include <vector>
#include <memory>
#include <time.h>


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

struct RubbishInfo{
	int box[4];
	
	float score;
	int cls;
};

class Read_MNN
{
	private:
		// 模型配置信息
		string input_tensor_name 	= "input_data"; // 输入节点
		string output_tensor_name0 	= "concat"; 	// 输出节点1
		string output_tensor_name1 	= "concat_1"; 	// 输出节点2
			
		std::shared_ptr<MNN::Interpreter> net;
		MNN::Session *session 		= nullptr;
		MNN::Tensor *input_tensor 	= nullptr;
		MNN::Tensor *tensor_boxes 	= nullptr;
		MNN::Tensor *tensor_score 	= nullptr;
		
		int input_shape[2];
		
		
	public:
		Read_MNN(const std::string &model_path, int threads=4, int precision = 2, int forward = MNN_FORWARD_CPU){
			/** 参数初始化，载入模型 **/
			net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
			if (nullptr == net) {
				cout << "load model failed." << endl;
			}
			
			MNN::ScheduleConfig config;
			config.numThread = threads;
			config.type      = static_cast<MNNForwardType>(forward);
			
			session = net->createSession(config);
			if (nullptr == session) {
				cout << "[LOG]create session failed." << endl;
			}
			input_tensor = net -> getSessionInput(session, input_tensor_name.c_str());
			/* 定义输出节点（未nms和socre阈值限定） */
			tensor_boxes  = net->getSessionOutput(session, output_tensor_name0.c_str());
			tensor_score  = net->getSessionOutput(session, output_tensor_name1.c_str());
			
		}
		~Read_MNN(){
			net -> releaseModel();
			net -> releaseSession(session);
		};
		
		Mat processImg(Mat img,int size);
		
		Mat saveImg(Mat img, vector<RubbishInfo> demo);
		
		int run(Mat &image, std::vector<RubbishInfo>* result, float score_thread=0.65, float nms_thread=0.45){
			/** 运行网络，预测图片 **/
			
			int shape[2] = {image.cols,image.rows};
			memcpy(input_shape,shape,sizeof(shape));
			image.convertTo(image, CV_32FC3);
			/* 定义tensor，数据载入内存 */
			std::vector<int> dims{1,input_shape[1], input_shape[0],3}; 
			auto nhwc_Tensor = MNN::Tensor::create<float>(dims, image.data, MNN::Tensor::TENSORFLOW);
			auto nhwc_data   = nhwc_Tensor->host<float>();
			auto nhwc_size   = nhwc_Tensor->size();
			::memcpy(nhwc_data, image.data, nhwc_size);
			/* 定义时，input维度为None，这里resize维度信息 */
			net->resizeTensor(input_tensor, {1, input_shape[1], input_shape[0], 3});
			net->resizeSession(session);
			/* 载入网络输入数据 */
			input_tensor->copyFromHostTensor(nhwc_Tensor);
			/* 运行模型 */
			net->runSession(session);
			
			/* 定义输出tensor */
			MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
			MNN::Tensor tensor_score_host(tensor_score, tensor_score->getDimensionType());
			/* 获取输出 */
			tensor_boxes->copyToHostTensor(&tensor_boxes_host);
			tensor_score->copyToHostTensor(&tensor_score_host);	
			
			int batch  = tensor_score_host.batch();
			int height = tensor_score_host.height();
			result->clear();
			std::vector<RubbishInfo> rubbish_tmp;
			int index;
			float confidence;
			float xmin_, ymin_, xmax_, ymax_;
			RubbishInfo rubb_info;
			for (int h=0; h<height; ++h){
				for (int w=0; w<batch; ++w){				
					index = h * batch + w;
					confidence = tensor_score_host.host<float>()[index];					
					if (confidence < score_thread){
						continue;
					}
					xmin_ = tensor_boxes_host.host<float>()[w];
					ymin_ = tensor_boxes_host.host<float>()[w+batch];
					xmax_ = tensor_boxes_host.host<float>()[w+batch*2];
					ymax_ = tensor_boxes_host.host<float>()[w+batch*3];
					//cout << "xmin: "<<xmin_ <<" ymin: "<< ymin_ <<" xmax: "<< xmax_ <<" ymax: "<< ymax_ << endl;
					rubb_info.score = confidence;
					rubb_info.box[0] = xmin_;
					rubb_info.box[1] = ymin_;
					rubb_info.box[2] = xmax_;
					rubb_info.box[3] = ymax_;
					rubb_info.cls = h;
					
					rubbish_tmp.push_back(rubb_info);
				}
				//cout <<"size: "<< rubbish_tmp.size() << endl;
				sort(rubbish_tmp.begin(),rubbish_tmp.end(),comp);
				nms(rubbish_tmp, result, nms_thread);
				rubbish_tmp.clear();
			}
			
			return 0;
		}
		
		static  bool comp(RubbishInfo a,RubbishInfo b){
			return a.score > b.score;
		}
		
		int nms(const std::vector<RubbishInfo>& inputs,std::vector<RubbishInfo>* result, float nms_thread){
			if (inputs.size() == 0){
				cout << "[LOG]inputs size is None " << endl;
				return -1;
			}
			int N = inputs.size();
			int pre_box[4];
			int cur_box[4];
			float iou_;
			std::vector<int> labels(N, -1);
			for (int i=0; i<N-1; ++i){	
				memcpy(pre_box,inputs[i].box,sizeof(inputs[i].box));
				for (int j=i+1; j<N; ++j){
					memcpy(cur_box,inputs[j].box,sizeof(inputs[j].box));
					iou_ = iou(pre_box, cur_box);
					if (iou_ < nms_thread){
						labels[j] = 0;
					}
				}
			}
			
			for (int i=0; i<N; ++i){
				if (labels[i] == -1){
					result->push_back(inputs[i]);
				}
			}

			return 0;
		}
		

		
		float iou(int box0[], int box1[]) 
		{
			float xmin0 = box0[0];
			float ymin0 = box0[1];
			float xmax0 = box0[2];
			float ymax0 = box0[3];
			
			float xmin1 = box1[0];
			float ymin1 = box1[1];
			float xmax1 = box1[2];
			float ymax1 = box1[3];

			float w = fmax(0.0f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
			float h = fmax(0.0f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));
			
			float i = w * h;
			float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
			
			if (u <= 0.0) return 0.0f;
			else          return i/u;
		}

};


Mat Read_MNN::processImg(Mat img,int size)
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

	
	return img2;
}

Mat Read_MNN::saveImg(Mat img, vector<RubbishInfo> demo){
	int N = demo.size();
	float xmin, ymin, xmax, ymax;
	float score_;
	string classes[3] = {"suliaoping","zhihe","yilaguan"};
	int index;
	string cls;
	for (int i=0; i<N; i++){
		xmin = demo[i].box[0];
		ymin = demo[i].box[1];
		xmax = demo[i].box[2];
		ymax = demo[i].box[3];
		
		xmin = fmax(0, floor(xmin + 0.5));
		ymin = fmax(0, floor(ymin + 0.5));
		xmax = fmin(input_shape[1], floor(xmax + 0.5));
		ymax = fmin(input_shape[0], floor(ymax + 0.5));
		score_ = demo[i].score;
		index = demo[i].cls;
		cls = classes[index];
		cout << "xmin: "<<xmin <<" ymin: "<< ymin <<" xmax: "<< xmax <<" ymax: "<< ymax << " score: " << score_ << " cls: " << cls << endl;
		rectangle(img,Point(xmin,ymin),Point(xmax,ymax),Scalar(255,0,0),2,8,0);
	}
	return img;
}

int main(){
	char const *filename = "/home/zsharp/sdk_develop/rubbish/data/image.jpg";
	Read_MNN model("/home/zsharp/sdk_develop/rubbish/data/tiny/tiny.mnn");
	Mat image;
	
	while (1){
		image = imread(filename);
		cvtColor(image,image,CV_BGR2RGB);
		clock_t start =	clock();
		std::vector<RubbishInfo> rubbish;
		model.run(image, &rubbish);
		cout << rubbish.size() << endl;
		Mat img;
		img = model.saveImg(image,rubbish);
		imwrite("./data/img_result.jpg",img);
		
		clock_t end = clock();
		double times = (double)(end-start)/CLOCKS_PER_SEC;
		cout <<"[LOG]Times: "<< times <<endl;
		break;
	}
	return 0;
}
//g++ test.cpp -o test `pkg-config --libs --cflags opencv,mnn` -ldl
