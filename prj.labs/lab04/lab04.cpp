#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <string>
#include "single_include/nlohmann/json.hpp"

int main() {
  using namespace nlohmann;
  const std::string pathToDataFolder = "../../../data/";
  const std::string videoPath = pathToDataFolder + "banknote5.mp4";
  cv::VideoCapture cap(videoPath);
  int frame_counter = 0;
  int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
  int fr = totalFrames / 5;
  const std::string folder = "vid5/";
  const std::string name = "Frame_" + std::to_string(fr);
  const std::string fileName1 = name + ".png";
  const std::string fileName2 = name + "_gray.png";
  const std::string fileName3 = name + "_thresh.png";
  cv::Mat frame;
  
  if (!cap.isOpened()) {
    std::cout << "Error opening video stream or file" << std::endl;
  }

  //Read json
  std::ifstream stream(folder + "labels.json");
  json j;
  stream >> j;

  //Iterate frame by frame
  while (true) {

    //Get frame
    cv::Mat tmpFrame;
    cap >> tmpFrame;
    if (tmpFrame.empty()) {
      std::cout << std::endl << frame_counter << std::endl;
      break;
    }

    //Process selected frame
    if (frame_counter % fr == 0 && frame_counter / fr >= 2 && frame_counter / fr <= 4) {

      tmpFrame.copyTo(frame);
      cv::imwrite(folder + std::to_string(frame_counter) + fileName1, frame);

      //Resize current frame
      cv::Mat resized(360, 640, CV_8UC1);
      cv::resize(frame, resized, cv::Size(360, 640), cv::INTER_NEAREST);
      
      //Extract data from json object
      std::string name = std::to_string(frame_counter) + fileName1;
      std::vector<int> x = j[name]["regions"][0]["shape_attributes"]["all_points_x"].get<std::vector<int>>();
      std::vector<int> y = j[name]["regions"][0]["shape_attributes"]["all_points_y"].get<std::vector<int>>();

      //Draw a polygon from json data
      std::vector<cv::Point> contour;
      for (int i = 0; i < x.size(); i++) 
        contour.push_back(cv::Point(x[i], y[i]));
      
      std::vector<std::vector<cv::Point> > allContours;
      allContours.push_back(contour);
      cv::Mat mask(1920, 1080, CV_8UC1);
      mask = 0;
      cv::fillPoly(mask, allContours, 255);

      //Resize mask
      cv::Mat maskResized(640, 360, CV_8UC1);
      cv::resize(mask, maskResized, cv::Size(360, 640), cv::INTER_NEAREST);

      //Convert to grayscale
      cv::Mat frameGray;
      cv::cvtColor(resized, frameGray, cv::COLOR_BGR2GRAY);

      //Apply Otsu thresh
      cv::Mat thresh;
      cv::threshold(frameGray, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

      //Check left upper and right lower corners and invert thresh
      if (thresh.at<uint8_t>(0, 0) == 255 && thresh.at<uint8_t>(thresh.rows - 1, thresh.cols - 1)) {
        cv::Mat inverted;
        inverted = cv::Scalar::all(255) - thresh;
        inverted.copyTo(thresh);
        std::cout << "\n INVERTED!\n";
      }
      
      //Get the largest connected component
      cv::Mat components;
      int nLabels = cv::connectedComponents(thresh, components, 8);
      std::vector<int> labelStats(nLabels, 0);
      for (int r = 0; r < thresh.rows; ++r) {
        for (int c = 0; c < thresh.cols; ++c) {
          int label = components.at<int>(r, c);
          labelStats[label]++;
        }
      }
      int maxLabel = -1;
      int maxLabelIndex = -1;
      for (int i = 1; i < labelStats.size(); i++) {
        if (labelStats[i] > maxLabel) {
          maxLabel = labelStats[i];
          maxLabelIndex = i;
        }
      }

      //Draw the largest component
      cv::Mat resMask(components.rows, components.cols, CV_8UC1);
      resMask = 0;
      for (int r = 0; r < resMask.rows; ++r) {
        for (int c = 0; c < resMask.cols; ++c) {
          if (components.at<int>(r, c) == maxLabelIndex) 
            resMask.at<uint8_t>(r, c) = 255;
        }
      }

      //Fill holes
      std::vector<std::vector<cv::Point> > contoursVector;
      cv::findContours(resMask, contoursVector, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
      cv::Mat filledMask(resMask.size(), CV_8UC1, cv::Scalar(0));
      for (int contourIndex = 0; contourIndex < contoursVector.size(); contourIndex++) {
        cv::drawContours(resMask, contoursVector, contourIndex, cv::Scalar(255), -1);
      }
     
      //Get main contour and draw min area rect
      std::vector<std::vector<cv::Point> > tmpContours;
      cv::findContours(resMask, tmpContours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
      cv::RotatedRect banknoteBox = cv::minAreaRect(tmpContours[0]);
      cv::Point2f boxPoints[4];
      banknoteBox.points(boxPoints);
      std::vector<std::vector<cv::Point>> boxVec(1);
      boxVec[0].push_back(boxPoints[0]);
      boxVec[0].push_back(boxPoints[1]);
      boxVec[0].push_back(boxPoints[2]);
      boxVec[0].push_back(boxPoints[3]);
      for (int i = 0; i < 4; i++) {
        std::cout << "\n" << boxVec[0][i].x << " , " << boxVec[0][i].y << "\n";
      }
      cv::fillPoly(filledMask, boxVec, cv::Scalar(255));
      
      //Save tmp results
      cv::imwrite(folder + std::to_string(frame_counter) + "TEST_FILLED.png", filledMask);
      cv::imwrite(folder + std::to_string(frame_counter) + "TEST_RESMASK.png", resMask);

      //Resize the result
      cv::Mat resizedResult(1080, 1920, CV_8UC1);
      cv::resize(filledMask, resizedResult, cv::Size(1080, 1920), cv::INTER_NEAREST);

      //Calculate IoU
      cv::Mat un = resizedResult + mask;
      cv::Mat inter = mask.mul(resizedResult);
      double normalize = 1920.0 * 1080.0;
      int unionPixels = 0;
      int intersectionPixels = 0;
      cv::Mat resColMask(resizedResult.size(), CV_8UC3);
      cv::Mat trueColMask(mask.size(), CV_8UC3);
      resColMask = cv::Scalar(0, 0, 0);
      trueColMask = cv::Scalar(0, 0, 0);
      for (int r = 0; r < resizedResult.rows; ++r) {
        for (int c = 0; c < resizedResult.cols; ++c) {
          if (un.at<uint8_t>(r, c) == 255) unionPixels++;
          if (inter.at<uint8_t>(r, c) == 255) intersectionPixels++;
          if (resizedResult.at<uint8_t>(r, c) == 255) resColMask.at<cv::Vec3b>(r, c)[2] = 255;
          if (mask.at<uint8_t>(r, c) == 255) trueColMask.at<cv::Vec3b>(r, c)[1] = 255;
        }
      }
      double iou = 1.0 * intersectionPixels / unionPixels;
      std::cout << "\n"<< iou << "\n";
  
      //Visualize
      cv::addWeighted(resColMask, 0.5, trueColMask, 0.5, 0.0, trueColMask);
      cv::addWeighted(trueColMask, 0.5, frame, 0.5, 0.0, frame);
      cv::imwrite(folder + "RES_" + std::to_string(iou) + "_" + std::to_string(frame_counter) + fileName1, frame);
    }
    frame_counter++;
  }
  cap.release();
}