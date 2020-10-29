// ROS libs
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>

#include <sensor_msgs/point_cloud2_iterator.h>

// Other dep libs
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>

// Native Libs
#include <string>
#include <cmath>
#include <chrono>

// Local Libs
#include "PreAllocator.h"

// ROS Node and Publishers
ros::NodeHandle * nh;
ros::NodeHandle * pnh;
ros::Publisher pc_pub;
ros::Publisher leftCamPub;
ros::Publisher rightCamPub;
ros::Publisher disparityPub;

// ROS Callbacks
void update_callback(const ros::TimerEvent&);

// ROS Params
double frequency;
int seq = 0;
int numDisparities;
int blockSize;
std::string camLeftPath;
std::string camRightPath;
std::string leftCamMatrixS;
std::string rightCamMatrixS;
std::string leftCamDistS;
std::string rightCamDistS;
std::string RmatS;
std::string TvecS;

// Global Vars
const cv::Size2i imSize(640,480);
const int MISSING_Z = 10000;

sensor_msgs::PointCloud2 pcmsg;

cv::VideoCapture * cameraLeft;
cv::VideoCapture * cameraRight;
cv::Ptr<cv::StereoBM> stereo;
cv::Mat leftCamMat(3, 3, CV_64FC1);
cv::Mat rightCamMat(3, 3, CV_64FC1);
cv::Mat leftCamDist(5, 1, CV_64FC1);
cv::Mat rightCamDist(5, 1, CV_64FC1);
cv::Mat Tvec(3, 1, CV_64FC1);
cv::Mat Rmat(3, 3, CV_64FC1);

cv::Mat Rleft;
cv::Mat Rright;
cv::Mat Pleft;
cv::Mat Pright;
cv::Mat Q;

cv::Mat mapL1;
cv::Mat mapL2;
cv::Mat mapR1;
cv::Mat mapR2;

cv::Mat frameLeft;
cv::Mat frameRight;
cv::Mat gsLeft;
cv::Mat gsRight;
cv::Mat rmgsLeft;
cv::Mat rmgsRight;
cv::Mat disparity;
cv::Mat ndisp;
cv::Mat points;
cv::Mat colors;

// Function defs
void parseCSVMat(std::string mat, cv::Mat * out);
void customReproject(const cv::Mat& disparity, const cv::Mat& Q, cv::Mat& colors, std::vector<uint8_t>& out);
inline bool isValidPoint(const cv::Vec3f& pt);

int main(int argc, char** argv) {
  // Init ROS
  ros::init(argc, argv, "stereocam_node");
  nh = new ros::NodeHandle();
  pnh = new ros::NodeHandle("~");

  // Params
  pnh->param<double>("frequency", frequency, 30.0);
  pnh->param<int>("numDisparities", numDisparities, 144);
  pnh->param<int>("blockSize", blockSize, 19);
  pnh->param<std::string>("leftCamMatrix", leftCamMatrixS, "");
  pnh->param<std::string>("rightCamMatrix", rightCamMatrixS, "");
  pnh->param<std::string>("leftCamDist", leftCamDistS, "");
  pnh->param<std::string>("rightCamDist", rightCamDistS, "");
  pnh->param<std::string>("Rmat", RmatS, "");
  pnh->param<std::string>("Tvec", TvecS, "");
  pnh->param<std::string>("camLeftPath", camLeftPath, "/dev/video0");
  pnh->param<std::string>("camRightPath", camRightPath, "/dev/video1");

  // Init OCV
  parseCSVMat(leftCamMatrixS, &leftCamMat);
  parseCSVMat(rightCamMatrixS, &rightCamMat);
  parseCSVMat(leftCamDistS, &leftCamDist);
  parseCSVMat(rightCamDistS, &rightCamDist);
  parseCSVMat(RmatS, &Rmat);
  parseCSVMat(TvecS, &Tvec);

  //cv::namedWindow("fuck");

  cv::stereoRectify(leftCamMat, leftCamDist, rightCamMat, rightCamDist, imSize, Rmat, Tvec, Rleft, Rright, Pleft, Pright, Q);
  cv::initUndistortRectifyMap(leftCamMat, leftCamDist, Rleft, Pleft, imSize, CV_8UC1, mapL1, mapL2);
  cv::initUndistortRectifyMap(rightCamMat, rightCamDist, Rright, Pright, imSize, CV_8UC1, mapR1, mapR2);
  Q.convertTo(Q, CV_32F);

  cameraLeft = new cv::VideoCapture(camLeftPath);
  cameraRight = new cv::VideoCapture(camRightPath);
  cameraLeft->set(cv::CAP_PROP_FPS, 30.0);
  cameraRight->set(cv::CAP_PROP_FPS, 30.0);
  stereo = cv::StereoBM::create(numDisparities, blockSize);

  pcmsg.is_bigendian = false;
  pcmsg.is_dense = false;
  pcmsg.point_step = 16;
  sensor_msgs::PointField pf;
  pf.name="x"; 
  pf.count=1; 
  pf.datatype=7; 
  pf.offset=0;
  pcmsg.fields.push_back(pf);
  pf.name="y";
  pf.offset=4;
  pcmsg.fields.push_back(pf);
  pf.name="z";
  pf.offset=8;
  pcmsg.fields.push_back(pf);
  pf.name="rgb";
  pf.offset=12;
  pcmsg.fields.push_back(pf);

  // Subscribers
  ros::Timer update_timer = nh->createTimer(ros::Duration(1.0/frequency), update_callback);

  // Publishers
  pc_pub = nh->advertise<sensor_msgs::PointCloud2>("stereocam/points", 10);
  leftCamPub = nh->advertise<sensor_msgs::Image>("stereocam/left", 1);
  rightCamPub = nh->advertise<sensor_msgs::Image>("stereocam/right", 1);
  disparityPub = nh->advertise<sensor_msgs::Image>("stereocam/disparity", 1);

  // Spin
  ros::spin();
}

void update_callback(const ros::TimerEvent&) {
  std_msgs::Header header;
  header.seq = seq++;
  header.stamp = ros::Time::now();
  header.frame_id = "stereocam";

  cameraLeft->grab();
  cameraRight->grab();

  auto start = std::chrono::high_resolution_clock::now(); 
  bool gotLeft = cameraLeft->retrieve(frameLeft);
  bool gotRight = cameraRight->retrieve(frameRight);

  if (!gotLeft || !gotRight) {
    std::cerr << "Failed to acquire frame" << std::endl;
    exit(-1);
  }

  // convert to grayscale
  cv::cvtColor(frameLeft, gsLeft, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frameRight, gsRight, cv::COLOR_BGR2GRAY);
  auto colortime = std::chrono::high_resolution_clock::now(); 
  // remap images
  cv::remap(gsLeft, rmgsLeft, mapL1, mapL2, cv::INTER_LINEAR);
  cv::remap(gsRight, rmgsRight, mapR1, mapR2, cv::INTER_LINEAR);
  // Convert color representation for pointcloud
  cv::remap(frameLeft, colors, mapL1, mapL2, cv::INTER_LINEAR);
  auto remaptime = std::chrono::high_resolution_clock::now(); 

  // zhu li do the thing
  stereo->compute(rmgsLeft, rmgsRight, disparity);
  auto stereotime = std::chrono::high_resolution_clock::now(); 

  // compute reprojection
  disparity.convertTo(disparity, CV_32F, 16.0);
  //cv::cvtColor(colors, colors, cv::COLOR_BGR2RGBA);
  //cv::reprojectImageTo3D(disparity, points, Q, true);

  pcmsg.header = header;
  pcmsg.height = disparity.rows;
  pcmsg.width = disparity.cols;
  pcmsg.row_step = pcmsg.width * pcmsg.point_step;
  int pointslen = sizeof(float) * 4 * pcmsg.width * pcmsg.height;
  std::vector<uint8_t> pointvec;
  pointvec.resize(pointslen, 0);

  customReproject(disparity, Q, colors, pointvec);
  auto reproj1time = std::chrono::high_resolution_clock::now();
  
  pcmsg.data = pointvec;

  auto reproj2time = std::chrono::high_resolution_clock::now();  
  auto colordur = std::chrono::duration_cast<std::chrono::milliseconds>(colortime - start);
  auto remapdur = std::chrono::duration_cast<std::chrono::milliseconds>(remaptime - start); 
  auto stereodur = std::chrono::duration_cast<std::chrono::milliseconds>(stereotime - start); 
  auto reproj1dur = std::chrono::duration_cast<std::chrono::milliseconds>(reproj1time - start); 
  auto reproj2dur = std::chrono::duration_cast<std::chrono::milliseconds>(reproj2time - start); 
  std::cout << "Cumulative times:" << std::endl;
  std::cout << "Color: " << colordur.count() << "ms" << std::endl;
  std::cout << "Remap: " << remapdur.count() << "ms" << std::endl;
  std::cout << "Stereo: " << stereodur.count() << "ms" << std::endl;
  std::cout << "ReprojectImageTo3D: " << reproj1dur.count() << "ms" << std::endl;
  std::cout << "Convert points: " << reproj2dur.count() << "ms" << std::endl << std::endl;

  // publish disparity
  cv::normalize(disparity, ndisp, 255, 0, cv::NORM_MINMAX, CV_8U);

  sensor_msgs::ImagePtr msgLeft = cv_bridge::CvImage(header, "bgr8", frameLeft).toImageMsg();
  sensor_msgs::ImagePtr msgRight = cv_bridge::CvImage(header, "bgr8", frameRight).toImageMsg();
  sensor_msgs::ImagePtr msgDisparity = cv_bridge::CvImage(header, "mono8", ndisp).toImageMsg();

  leftCamPub.publish(msgLeft);
  rightCamPub.publish(msgRight);
  disparityPub.publish(msgDisparity);
  pc_pub.publish(pcmsg);
}

void parseCSVMat(std::string mat, cv::Mat * out) {
  size_t pos = 0;
  size_t posinner = 0;
  int row = 0;
  int col = 0;
  while ((pos = mat.find("\n")) != std::string::npos) {
    std::string line = mat.substr(0, pos);
    col = 0;
    while ((posinner = line.find(",")) != std::string::npos) {
      std::string item = line.substr(0, posinner);
      out->at<double>(row, col++) = std::stod(item);
      line.erase(0, posinner + 1);
    }
    // Last item in row
    out->at<double>(row, col++) = std::stod(line);
    mat.erase(0, pos + 1);
    row++;
  }
}

inline bool isValidPoint(const cv::Vec3f& pt)
{
  return pt[2] != MISSING_Z && !std::isinf(pt[2]);
}

void customReproject(const cv::Mat& disparity, const cv::Mat& Q, cv::Mat& colors, std::vector<uint8_t>& out)
{
	CV_Assert(disparity.type() == CV_32F && !disparity.empty());
	CV_Assert(Q.type() == CV_32F && Q.cols == 4 && Q.rows == 4);
  uint8_t* it = out.data();

	// Getting the interesting parameters from Q, everything else is zero or one
	float Q03 = Q.at<float>(0, 3);
	float Q13 = Q.at<float>(1, 3);
	float Q23 = Q.at<float>(2, 3);
	float Q32 = Q.at<float>(3, 2);
	float Q33 = Q.at<float>(3, 3);

	for (int i = 0; i < disparity.rows; i++)
	{
		const float* disp_ptr = disparity.ptr<float>(i);

		for (int j = 0; j < disparity.cols; j++)
		{
			const float pw = 1.0f / (disp_ptr[j] * Q32 + Q33);
			float point[4];
      
      point[0] = (static_cast<float>(j)+Q03) * pw;
			point[1] = (static_cast<float>(i)+Q13) * pw;
			point[2] = Q23 * pw;
      point[3] = *reinterpret_cast<float*>(colors.ptr(i,j));
      std::memcpy(it, point, 16);
      it += 16;
		}
	}
}