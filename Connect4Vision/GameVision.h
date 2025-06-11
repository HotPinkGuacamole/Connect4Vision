// GameVision.h

#ifndef GAME_VISION_H
#define GAME_VISION_H

#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <iostream>

constexpr int ROWS = 6;
constexpr int COLS = 7;
using Board = std::array<std::array<int, COLS>, ROWS>;

class GameVision {
public:
    GameVision();
    void calibrateHSV();
    bool detectAndWarpBoard(const cv::Mat& frame, cv::Mat& warped);
    void detectDiscs(const cv::Mat& warped, Board& board);
    static bool boardsDiffer(const Board& a, const Board& b);
    static void printBoard(const Board& board);

private:
    // ROOD/ROZE (2 hue-ranges)
    int hLowRed1_, hHighRed1_;
    int hLowRed2_, hHighRed2_;
    int sLowRed_;
    int vLowRed_;

    // GROEN
    int hLowGreen_, hHighGreen_;
    int sLowGreen_;
    int vLowGreen_;

    // BORD (goud/oranje)
    int hLowBoard_, hHighBoard_;
    int sLowBoard_;
    int vLowBoard_;

    cv::Mat perspectiveM_;
    cv::Size warpSize_;

    void initDefaultHSV();
    void createHSVTrackbars();
    int classifyColor(const cv::Mat& roiHSV);
};

#endif // GAME_VISION_H
