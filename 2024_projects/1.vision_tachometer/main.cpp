/**
 * Author: enpeicv
 * Date: 2024-01-10
 * vision based tachometer
 **/

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <iostream>

class VideoProcessor
{
private:
    cv::VideoCapture capture;      // video capture object
    cv::VideoWriter writer;        // video writer object
    cv::Mat reference_img;         // reference image
    std::vector<double> data_list; // similarity data list
    float resize_factor = 0.5;     // resize factor

public:
    VideoProcessor(char *filename, char *reference_img);                                    // constructor
    ~VideoProcessor();                                                                      // destructor
    void processvideo();                                                                    // process video
    double calculateSimilarity(cv::Mat image1, cv::Mat image2, cv::Mat &matchImage);        // calculate similarity
    bool drawGraph(cv::Mat &bg);                                                            // draw graph
    void findPeaks(std::vector<double> data, std::vector<double> &peaks, double threshold); // find peaks
};

VideoProcessor::VideoProcessor(char *filename, char *reference_file)
{
    // read video file
    capture.open(filename);
    // set height and width
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 480);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // get frame rate
    int rate = capture.get(cv::CAP_PROP_FPS);
    // cout fps
    std::cout << "fps: " << rate << std::endl;
    // check if video successfully opened
    if (!capture.isOpened())
    {
        std::cout << "error reading video file" << std::endl;
    }

    // read reference image in grayscale
    reference_img = cv::imread(reference_file, cv::IMREAD_GRAYSCALE);
    // resize reference image to half size to speed up
    // cv::resize(reference_img, reference_img, cv::Size(), resize_factor, resize_factor);
    // check if reference image successfully opened
    if (reference_img.empty())
    {
        std::cout << "error reading reference image" << std::endl;
    }
    // cout reference image size
    std::cout << "reference image size: " << reference_img.size().width << " x " << reference_img.size().height << std::endl;
}

VideoProcessor::~VideoProcessor()
{
    // release video capture object
    capture.release();
    // release video writer object
    writer.release();
}
void VideoProcessor::processvideo()
{
    // read video file
    auto last_peak_time = std::chrono::high_resolution_clock::now();
    std::string rpm_text;
    while (true)
    {
        // start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        capture >> frame;
        if (frame.empty())
        {
            break;
        }
        // convert frame to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // resize frame to half size to speed up
        cv::resize(gray, gray, cv::Size(), resize_factor, resize_factor);
        // cout frame size
        // std::cout << "frame size: " << gray.size().width << " x " << gray.size().height << std::endl;
        // calculate similarity
        cv::Mat matchImage;
        double similarity = calculateSimilarity(reference_img, gray, matchImage);
        // cout similarity
        // std::cout << "similarity: " << similarity << std::endl;
        // add similarity to data list
        data_list.push_back(similarity);
        // create a 800x800x3 black image
        cv::Mat bg = cv::Mat::zeros(800, 800, CV_8UC3);
        // draw graph
        auto new_revolution = drawGraph(bg);
        if (new_revolution)
        {
            // calculate time difference
            auto current_time = std::chrono::high_resolution_clock::now();
            auto revolution_duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_peak_time).count();
            // rpm
            double rpm = 60000000 / revolution_duration;
            // cout rpm
            std::cout << "rpm: " << rpm << std::endl;
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << rpm;
            std::string s = stream.str();
            rpm_text = "rpm: " + s;

            // reset last peak time
            last_peak_time = current_time;
        }

        // fps
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        double fps = 1000000 / duration;

        // put fps on frame
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << fps;
        std::string s = stream.str();
        s = "fps: " + s;
        cv::putText(bg, s, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        // put rpm on frame
        cv::putText(bg, rpm_text, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // horizontal merge frame and match image
        cv::Mat horizontal_merge;
        // convert gray to 3 channels
        cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
        // horizontal merge
        cv::hconcat(gray, matchImage, horizontal_merge);

        // show frame
        cv::imshow("frame", horizontal_merge);

        cv::imshow("graph", bg);
        // wait for key press
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }
}
double VideoProcessor::calculateSimilarity(cv::Mat image1, cv::Mat image2, cv::Mat &matchImage)
{
    // Detect SIFT keypoints and descriptors
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    sift->detectAndCompute(image1, cv::Mat(), keypoints1, descriptors1);
    sift->detectAndCompute(image2, cv::Mat(), keypoints2, descriptors2);

    // Use FLANN matcher
    cv::FlannBasedMatcher flann;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    flann.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // Filter good matches
    std::vector<cv::DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); ++i)
    {
        if (knnMatches[i][0].distance < 0.7 * knnMatches[i][1].distance)
        {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    // Calculate similarity
    double similarity = static_cast<double>(goodMatches.size()) / std::max(keypoints1.size(), keypoints2.size());

    // Draw matches
    cv::drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, matchImage);

    return similarity;
}

bool VideoProcessor::drawGraph(cv::Mat &bg)
{
    bool find_new_revolution = false;
    // draw line
    auto line1_left_p = cv::Point(0, 400);
    auto line1_right_p = cv::Point(800, 400);
    auto line2_left_p = cv::Point(400, 0);
    auto line2_right_p = cv::Point(400, 800);
    cv::line(bg, line1_left_p, line1_right_p, cv::Scalar(0, 255, 0), 3);
    cv::line(bg, line2_left_p, line2_right_p, cv::Scalar(0, 255, 0), 3);

    // iterate through data list
    // get last 1000 data of data list
    std::vector<double> data_list_1000;
    if (data_list.size() < 1000)
    {
        data_list_1000 = data_list;
    }
    else
    {
        data_list_1000 = std::vector<double>(data_list.end() - 1000, data_list.end());
    }
    // reverse data list
    std::reverse(data_list_1000.begin(), data_list_1000.end());
    std::vector<cv::Point> polylines_points;

    for (int i = 0; i < data_list_1000.size(); i++)
    {
        // draw point
        auto point = cv::Point(700 + (-1 * i), 400 - (data_list_1000[i] * 300));
        polylines_points.push_back(point);
        if (i == 0)
        {
            cv::circle(bg, point, 10, cv::Scalar(255, 0, 255), -1);
            // round data_list_1000[i] to 2 decimal places
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << data_list_1000[i];
            std::string s = stream.str();

            cv::putText(bg, s, cv::Point(point.x + 20, point.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
        }
    }
    // draw polylines
    cv::polylines(bg, polylines_points, false, cv::Scalar(255, 0, 255), 2);

    // find peaks
    std::vector<double> peaks;
    findPeaks(data_list_1000, peaks, 0.2);

    // draw peaks
    for (int i = 0; i < peaks.size(); i++)
    {
        auto index = peaks[i];
        if (index == 1)
            find_new_revolution = true;
        // std::cout << "index: " << index << std::endl;
        auto point = cv::Point(700 + (-1 * index), 400 - (data_list_1000[index] * 300));
        cv::circle(bg, point, 10, cv::Scalar(0, 255, 0), -1);
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << data_list_1000[index];
        std::string s = stream.str();
        cv::putText(bg, s, cv::Point(point.x, point.y - 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    return find_new_revolution;
}

void VideoProcessor::findPeaks(std::vector<double> data, std::vector<double> &peaks, double threshold)
{
    for (int i = 1; i < data.size() - 1; i++)
    {
        
        if (data[i] > data[i - 1] && data[i] > data[i -2] && data[i] > data[i + 1] && data[i] > data[i + 2] && data[i] > threshold)
        {
            peaks.push_back(i);
        }
    }
}

int main(int argc, char **argv)
{
    // check arguments
    if (argc != 3)
    {
        std::cout << "usage: ./build/HelloWorld <video_file> <reference_image>" << std::endl;
        return -1;
    }
    // initialize video processor
    char *filename = argv[1];
    char *reference_file = argv[2];
    VideoProcessor processor(filename, reference_file);
    // process video
    processor.processvideo();

    return 0;
}
