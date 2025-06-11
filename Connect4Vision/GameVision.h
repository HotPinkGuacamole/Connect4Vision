#ifndef GAMEVISION_H
#define GAMEVISION_H

#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

constexpr int ROWS = 6;
constexpr int COLS = 7;
using Board = array<array<int, COLS>, ROWS>;

class GameVision {
public:
    GameVision(int cameraIndex = 0);
    ~GameVision() noexcept;

    bool detectAndWarpBoard(const Mat& frame, Mat& warped);
    void detectDiscs(const Mat& warped, Board& board);

    void updateBoard(const Board& board);
    bool tick();
    Board getState() const;

    static bool boardsDiffer(const Board& a, const Board& b);
    static void printBoard(const Board& board);

    void createSliderWindow();

private:
    int classifyColor(const Mat& roiHSV);

    VideoCapture cap_;
    Mat           perspectiveM_;
    Size          warpSize_{ 700,600 };

    Board oldBoard_{}, newBoard_{};
    steady_clock::time_point changeStart_;
    bool firstTick_{ true };

    int hLowRed1_, hHighRed1_, hLowRed2_, hHighRed2_, sLowRed_, vLowRed_;
    int hLowBlue_, hHighBlue_, sLowBlue_, vLowBlue_;
    int hLowBoard_, hHighBoard_, sLowBoard_, vLowBoard_;
};

#endif // GAMEVISION_H
