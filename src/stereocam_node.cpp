// ROS libs
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

// Other dep libs
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/ocl.hpp>

// Native Libs
#include <string>
#include <cmath>
#include <chrono>
#include <thread> 
#include <mutex> 

// ROS Node and Publishers
ros::NodeHandle *nh;
ros::NodeHandle *pnh;
ros::Publisher pc_pub;
ros::Publisher leftCamPub;
ros::Publisher rightCamPub;
ros::Publisher disparityPub;

// ROS Callbacks
void update_callback(const ros::TimerEvent &);

// ROS Params
double frequency;
int seq = 0;
int numDisparities;
int minDisparity;
int uniquenessRatio;
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
const cv::Size2i imSize(640, 480);
const int MISSING_Z = 10000;

sensor_msgs::PointCloud2 pcmsg;

cv::VideoCapture *cameraLeft;
cv::VideoCapture *cameraRight;
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
cv::Mat colors;

std::mutex frameLeftMtx;
std::mutex frameRightMtx;
std::mutex disparityMtx;
std::mutex pcMtx;

// Function defs
void parseCSVMat(std::string mat, cv::Mat *out);
void customReproject(const cv::Mat &disparity, const cv::Mat &Q, cv::Mat &colors, std::vector<uint8_t> &out);
inline bool isValidPoint(const cv::Vec3f &pt);
void toImageMsg(cv::Mat& image, std_msgs::Header header, const std::string& encoding, sensor_msgs::Image& ros_image);
void delegatePublishImg(cv::Mat* img, std::string encoding, std_msgs::Header* header, ros::Publisher* pub, std::mutex* mtx);
void delegatePublishDisparity(cv::Mat* img, std::string encoding, std_msgs::Header* header, ros::Publisher* pub, std::mutex* mtx);
void delegatePublishPC(sensor_msgs::PointCloud2* pcmsg, std_msgs::Header* header, ros::Publisher* pub, std::mutex* mtx);

int main(int argc, char **argv)
{
    cv::ocl::setUseOpenCL(true);
    // Init ROS
    ros::init(argc, argv, "stereocam_node");
    nh = new ros::NodeHandle();
    pnh = new ros::NodeHandle("~");

    // Params
    pnh->param<double>("frequency", frequency, 30.0);
    pnh->param<int>("numDisparities", numDisparities, 144);
    pnh->param<int>("minDisparity", minDisparity, 0);
    pnh->param<int>("blockSize", blockSize, 19);
    pnh->param<int>("uniquenessRatio", uniquenessRatio, 0);
    pnh->param<std::string>("leftCamMatrix", leftCamMatrixS, "");
    pnh->param<std::string>("rightCamMatrix", rightCamMatrixS, "");
    pnh->param<std::string>("leftCamDist", leftCamDistS, "");
    pnh->param<std::string>("rightCamDist", rightCamDistS, "");
    pnh->param<std::string>("Rmat", RmatS, "");
    pnh->param<std::string>("Tvec", TvecS, "");
    pnh->param<std::string>("camLeftPath", camLeftPath, "/dev/video0");
    pnh->param<std::string>("camRightPath", camRightPath, "/dev/video1");

    // Read calibration matrices
    parseCSVMat(leftCamMatrixS, &leftCamMat);
    parseCSVMat(rightCamMatrixS, &rightCamMat);
    parseCSVMat(leftCamDistS, &leftCamDist);
    parseCSVMat(rightCamDistS, &rightCamDist);
    parseCSVMat(RmatS, &Rmat);
    parseCSVMat(TvecS, &Tvec);

    // Initialize undistortion and reprojection matrices
    cv::stereoRectify(leftCamMat, leftCamDist, rightCamMat, rightCamDist, imSize, Rmat, Tvec, Rleft, Rright, Pleft, Pright, Q);
    cv::initUndistortRectifyMap(leftCamMat, leftCamDist, Rleft, Pleft, imSize, CV_8UC1, mapL1, mapL2);
    cv::initUndistortRectifyMap(rightCamMat, rightCamDist, Rright, Pright, imSize, CV_8UC1, mapR1, mapR2);
    Q.convertTo(Q, CV_32F);

    // Set up video capture and stereo objects
    cameraLeft = new cv::VideoCapture(camLeftPath);
    cameraRight = new cv::VideoCapture(camRightPath);
    cameraLeft->set(cv::CAP_PROP_FPS, 30.0);
    cameraRight->set(cv::CAP_PROP_FPS, 30.0);
    stereo = cv::StereoBM::create(numDisparities, blockSize);
    stereo->setMinDisparity(minDisparity);
    //stereo->setUniquenessRatio(uniquenessRatio);
    stereo->setPreFilterType(1);
    stereo->setPreFilterCap(9);

    // Initialize pointcloud message, this stays the same every loop
    pcmsg.is_bigendian = false;
    pcmsg.is_dense = false;
    pcmsg.point_step = 16;
    sensor_msgs::PointField pf;
    pf.name = "x";
    pf.count = 1;
    pf.datatype = 7;
    pf.offset = 0;
    pcmsg.fields.push_back(pf);
    pf.name = "y";
    pf.offset = 4;
    pcmsg.fields.push_back(pf);
    pf.name = "z";
    pf.offset = 8;
    pcmsg.fields.push_back(pf);
    pf.name = "rgb";
    pf.offset = 12;
    pcmsg.fields.push_back(pf);

    // Subscribers
    ros::Timer update_timer = nh->createTimer(ros::Duration(1.0 / frequency), update_callback);

    // Publishers
    pc_pub = nh->advertise<sensor_msgs::PointCloud2>("stereocam/points", 10);
    leftCamPub = nh->advertise<sensor_msgs::Image>("stereocam/left/image_raw", 1);
    rightCamPub = nh->advertise<sensor_msgs::Image>("stereocam/right/image_raw", 1);
    disparityPub = nh->advertise<sensor_msgs::Image>("stereocam/disparity", 1);

    // Spin
    ros::spin();
}

void update_callback(const ros::TimerEvent &)
{
    std_msgs::Header header;
    header.seq = seq++;
    header.stamp = ros::Time::now();
    header.frame_id = "stereocam";

    cameraLeft->grab();
    cameraRight->grab();

    auto start = std::chrono::high_resolution_clock::now();
    frameLeftMtx.lock();
    frameRightMtx.lock();
    bool gotLeft = cameraLeft->retrieve(frameLeft);
    bool gotRight = cameraRight->retrieve(frameRight);
    frameLeftMtx.unlock();
    frameRightMtx.unlock();
    std::thread pubLeft(delegatePublishImg, &frameLeft, "bgr8", &header, &leftCamPub, &frameLeftMtx);
    pubLeft.detach();
    std::thread pubRight(delegatePublishImg, &frameRight, "bgr8", &header, &rightCamPub, &frameRightMtx);
    pubRight.detach();

    if (!gotLeft || !gotRight)
    {
        std::cerr << "Failed to acquire frame" << std::endl;
        exit(-1);
    }
    auto grabtime = std::chrono::high_resolution_clock::now();

    // convert to grayscale and remap
    // It's in this order so we only need to do two remaps, and also remap fewer channels
    cv::remap(frameLeft, colors, mapL1, mapL2, cv::INTER_LINEAR);
    cv::cvtColor(colors, gsLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frameRight, gsRight, cv::COLOR_BGR2GRAY);
    cv::remap(gsRight, rmgsRight, mapR1, mapR2, cv::INTER_LINEAR);
    
    auto remaptime = std::chrono::high_resolution_clock::now();

    // calculate disparity map
    disparityMtx.lock();
    stereo->compute(gsLeft, rmgsRight, disparity);
    auto stereotime = std::chrono::high_resolution_clock::now();

    disparity.convertTo(disparity, CV_32F, 16.0);
    disparityMtx.unlock();
    std::thread pubDisp(delegatePublishDisparity, &disparity, "mono8", &header, &disparityPub, &disparityMtx);
    pubDisp.detach();

    // Build the pointcloud message
    pcMtx.lock();
    pcmsg.height = disparity.rows;
    pcmsg.width = disparity.cols;
    pcmsg.row_step = pcmsg.width * pcmsg.point_step;
    int pointslen = sizeof(float) * 4 * pcmsg.width * pcmsg.height;
    pcmsg.data.resize(pointslen, 0);

    // Custom reproject function puts data directly into the pointcloud vector
    // saving about 200ms of copying again
    customReproject(disparity, Q, colors, pcmsg.data);
    pcMtx.unlock();
    // Publish pointcloud on new thread
    std::thread pubPC(delegatePublishPC, &pcmsg, &header, &pc_pub, &pcMtx);
    pubPC.detach();
    auto reprojtime = std::chrono::high_resolution_clock::now();

    auto grabdur = std::chrono::duration_cast<std::chrono::milliseconds>(grabtime - start);
    auto remapdur = std::chrono::duration_cast<std::chrono::milliseconds>(remaptime - start);
    auto stereodur = std::chrono::duration_cast<std::chrono::milliseconds>(stereotime - start);
    auto reprojdur = std::chrono::duration_cast<std::chrono::milliseconds>(reprojtime - start);
    std::cout << "Cumulative times:" << std::endl;
    std::cout << "Grab frames: " << grabdur.count() << "ms" << std::endl;
    std::cout << "Remap and color: " << remapdur.count() << "ms" << std::endl;
    std::cout << "Stereo: " << stereodur.count() << "ms" << std::endl;
    std::cout << "ReprojectImageTo3D: " << reprojdur.count() << "ms" << std::endl << std::endl;
}

void parseCSVMat(std::string mat, cv::Mat *out)
{
    size_t pos = 0;
    size_t posinner = 0;
    int row = 0;
    int col = 0;
    while ((pos = mat.find("\n")) != std::string::npos)
    {
        std::string line = mat.substr(0, pos);
        col = 0;
        while ((posinner = line.find(",")) != std::string::npos)
        {
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

inline bool isValidPoint(const cv::Vec3f &pt)
{
    return pt[2] != MISSING_Z && !std::isinf(pt[2]);
}

void customReproject(const cv::Mat &disparity, const cv::Mat &Q, cv::Mat &colors, std::vector<uint8_t> &out)
{
    CV_Assert(disparity.type() == CV_32F && !disparity.empty());
    CV_Assert(Q.type() == CV_32F && Q.cols == 4 && Q.rows == 4);
    uint8_t *it = out.data();

    // Getting the interesting parameters from Q, everything else is zero or one
    float Q03 = Q.at<float>(0, 3);
    float Q13 = Q.at<float>(1, 3);
    float Q23 = Q.at<float>(2, 3);
    float Q32 = Q.at<float>(3, 2);
    float Q33 = Q.at<float>(3, 3);

    for (int i = 0; i < disparity.rows; i++)
    {
        const float *disp_ptr = disparity.ptr<float>(i);

        for (int j = 0; j < disparity.cols; j++)
        {
            const float pw = 1.0f / (disp_ptr[j] * Q32 + Q33);
            float point[4];

            point[0] = (static_cast<float>(j) + Q03) * pw;
            point[1] = (static_cast<float>(i) + Q13) * pw;
            point[2] = Q23 * pw;
            point[3] = *reinterpret_cast<float *>(colors.ptr(i, j));
            std::memcpy(it, point, 16);
            it += 16;
        }
    }
}

void toImageMsg(cv::Mat& image, std_msgs::Header header, const std::string& encoding, sensor_msgs::Image& ros_image)
{
  ros_image.header = header;
  ros_image.height = image.rows;
  ros_image.width = image.cols;
  ros_image.encoding = encoding;
  ros_image.is_bigendian = false;
  ros_image.step = image.cols * image.elemSize();
  size_t size = ros_image.step * image.rows;
  ros_image.data.resize(size);

  if (image.isContinuous())
  {
    memcpy((char*)(&ros_image.data[0]), image.data, size);
  }
  else
  {
    // Copy by row by row
    uint8_t* ros_data_ptr = (uint8_t*)(&ros_image.data[0]);
    uint8_t* cv_data_ptr = image.data;
    for (int i = 0; i < image.rows; ++i)
    {
      memcpy(ros_data_ptr, cv_data_ptr, ros_image.step);
      ros_data_ptr += ros_image.step;
      cv_data_ptr += image.step;
    }
  }
}

void delegatePublishImg(cv::Mat* img, std::string encoding, std_msgs::Header* header, ros::Publisher* pub, std::mutex* mtx) {
    mtx->lock(); // blocking lock, in theory shouldn't interact with anything else because of timing
    sensor_msgs::Image msg;
    toImageMsg(*img, *header, encoding, msg);
    pub->publish(msg);
    mtx->unlock();
}

void delegatePublishDisparity(cv::Mat* img, std::string encoding, std_msgs::Header* header, ros::Publisher* pub, std::mutex* mtx) {
    mtx->lock(); // blocking lock, in theory shouldn't interact with anything else because of timing
    cv::normalize(*img, ndisp, 255, 0, cv::NORM_MINMAX, CV_8U);
    sensor_msgs::Image msg;
    toImageMsg(ndisp, *header, encoding, msg);
    pub->publish(msg);
    mtx->unlock();
}

void delegatePublishPC(sensor_msgs::PointCloud2* pcmsg, std_msgs::Header* header, ros::Publisher* pub, std::mutex* mtx) {
    mtx->lock();
    pcmsg->header = *header;
    pub->publish(*pcmsg);
    mtx->unlock();
}