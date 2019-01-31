// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include "dlib/image_processing/frontal_face_detector.h"
#include <dlib/opencv/cv_image.h>

#include "opencv2/opencv.hpp"
#include "timer_ticToc.h"

//#include "QtCore/QDirIterator"
//#include "QtCore/QJsonArray"

#include "QtCore/QtCore"

#include"cnpy.h"

using namespace dlib;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

dlib::matrix<float, 0, 1> process_job(const dlib::cv_image<dlib::bgr_pixel>& img_dlib, const dlib::rectangle& det, const dlib::shape_predictor& sp, anet_type& net)
{
	dlib::matrix<dlib::rgb_pixel> face_img_aligned;
	auto shape = sp(img_dlib, det);
	auto chip_details = get_face_chip_details(shape, 150, 0.25);
	extract_image_chip(img_dlib, chip_details, face_img_aligned);
	dlib::matrix<float, 0, 1> fvec_dlib = net(face_img_aligned);
	return fvec_dlib;
}


using namespace std;

int main(int argc, char* args[])
{		
	//QFile file;
	//file.setFileName("test_output.json");
	//file.open(QIODevice::WriteOnly | QIODevice::Text);

	//QJsonObject outer_dict;
		
	QStringList extensions = QStringList() << "*.jpg" << "*.jpeg" << "*.tiff" << "*.png" << "*.tif" << "*.bmp";

	/*
	calling convention:
	program.exe fpath_to_numpy_matrix_containing_feats fpath_to_numpy_matrix_containing_detections fpath_to_image_paths directory_to_extract
	*/

	char* fpath_to_numpy_matrix_feats = args[1];
	char* fpath_to_numpy_matrix_dets = args[2];
	char* fpath_to_file_img_paths = args[3];

	size_t ndims = 128;
	bool firstFeat = true;

	FILE* fid = fopen(fpath_to_file_img_paths, "w");
	if (fid == NULL)
	{
		printf("Cannot open file: %s\n", fid);
		exit(1);
	}

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor sp;
	anet_type net;
	
	try
	{
		dlib::deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
		dlib::deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
	}

	catch (const std::exception &exc)
	{
		std::cerr << exc.what();
	}

	unsigned int total_num_feats = 0;
 
	timer_ticToc tt;
	tt.tic();

	for (size_t i = 4; i < argc; i++)
	{
		std::cout << "Processing: " << args[i] << std::endl;
		QDir root_cur(args[i]);
		QDirIterator it(args[i], extensions, QDir::Files, QDirIterator::Subdirectories);
		while (it.hasNext())
		{
			QString fpath_img = it.next();
			std::string temp = root_cur.relativeFilePath(fpath_img).toStdString() + "\n";
			//printf("Processing inside root folder:  %s", temp.c_str());
									
			cv::Mat frame = cv::imread(fpath_img.toStdString());
			dlib::cv_image<dlib::bgr_pixel> img_dlib(frame);
			std::vector<dlib::rectangle> dets = detector(img_dlib);

			//printf("ndets = %d\n", dets.size());

			if (dets.size() != 1)
			{
				//printf("dets.size() != 1. Skipping\n");
				continue;
			}

			fputs(temp.c_str(), fid);
			
			dlib::rectangle det_rect = dets[0];
			dlib::matrix<float, 0, 1> fvec_dlib = process_job(img_dlib, det_rect, sp, net);

			std::vector<int> det_rect_vec({ (int)det_rect.left(), (int)det_rect.top(), (int)det_rect.width(), (int)det_rect.height() });
			
			std::vector<float> fvec(fvec_dlib.begin(), fvec_dlib.end());
			
			std::string save_mode;

			if (firstFeat)
			{
				save_mode = "w";
				firstFeat = false;
			}
			else
			{
				save_mode = "a";
			}
			
			cnpy::npy_save(fpath_to_numpy_matrix_dets, det_rect_vec.data(), { 1, 4 }, save_mode);
			cnpy::npy_save(fpath_to_numpy_matrix_feats, fvec.data(), { 1, ndims }, save_mode);

			total_num_feats++;
			
		}
	
		std::cout << "================================" << std::endl;
	}

	fclose(fid);

	printf("The entire operation took %f secs\n", tt.toc());
	printf("Total number of feature vectors extracted = %d\n", total_num_feats);


	/*file.write(QJsonDocument(outer_dict).toJson(QJsonDocument::Indented));
	file.close();*/

	//double myVar1 = 1.2;
	//char myVar2 = 'a';
	//cnpy::npz_save("out.npz", "myVar1", &myVar1, { 1 }, "w"); //"w" overwrites any existing file
	//cnpy::npz_save("out.npz", "myVar2", &myVar2, { 1 }, "a"); //"a" appends to the file we created above
	//cnpy::npz_save("out.npz", "myVar1", &myVar1, { 1 }, "a"); //"w" overwrites any existing file
	//myVar2 = 'b';
	//cnpy::npz_save("out.npz", "myVar2", &myVar2, { 1 }, "a"); //"a" appends to the file we created above
	//	
		

	
	//cnpy::npy_save("arr1.npy", vec.data(), { 1, nelem}, "w");
	//cnpy::npy_save("arr1.npy", vec2.data(), { 1, nelem }, "a");
	//cnpy::npy_save("arr1.npy", vec3.data(), { 1, nelem }, "a");

	//dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	//dlib::shape_predictor sp;
	//anet_type net;
	//
	//try
	//{
	//	dlib::deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
	//	dlib::deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
	//}

	//catch (const std::exception &exc)
	//{
	//	std::cerr << exc.what();
	//}
	//
	//cv::Mat frame;

	//int count_nfaces = 0;
	//double total_time = 0;

	//timer_ticToc tt;

	//while (true)
	//{
	//	capture >> frame;
	//	if (frame.empty())
	//	{
	//		break;
	//	}
	//	dlib::cv_image<dlib::bgr_pixel> img_dlib(frame);
	//	std::vector<dlib::rectangle> dets = detector(img_dlib);
	//	
	//	int ndets = dets.size();

	//	std::vector<dlib::matrix<float, 0, 1>> fvecs(ndets);

	//	count_nfaces += ndets;

	//	for (int i = 0; i < ndets; i++)
	//	{
	//		tt.tic();
	//		fvecs[i] = process_job(img_dlib, dets[i], sp, net);
	//		total_time += tt.toc();
	//	}			

	//	for (int i = 0; i < ndets; i++)
	//	{
	//		dlib::rectangle r_dlib = dets[i];
	//		cv::Rect r(r_dlib.left(), r_dlib.top(), r_dlib.width(), r_dlib.height());
	//		cv::rectangle(frame, r, cv::Scalar(255, 0, 0));
	//	}

	//	cv::imshow("win1", frame);
	//	cv::waitKey(1);			
	//}

	//printf("Average time taken for each feature extraction = %f secs", total_time / count_nfaces);


}