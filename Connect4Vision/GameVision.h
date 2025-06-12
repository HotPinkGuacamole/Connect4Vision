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
    /**
     * Constructor: opent de camera op index cameraIndex,
     * en zet default HSV-ranges klaar.
     */
    GameVision(int cameraIndex = 0);

    /** Destructor: sluit alle OpenCV-vensters */
    ~GameVision() noexcept;

    /**
     * detectAndWarpBoard:
     * - filtert op bordkleur (goud/oranje)
     * - doet morfologische ops om ruis te verwijderen
     * - zoekt grootste contour die een vierhoek is
     * - maakt een perspective warp naar 700×600
     * returnt true als warp gelukt is, anders false
     */
    bool detectAndWarpBoard(const Mat& frame, Mat& warped);

    /**
     * detectDiscs:
     * - splitst warped image in 6×7 grid
     * - voor elke cel een kleine HSV-ROI nemen
     * - roept classifyColor aan om 0/1/2 te krijgen
     * - tekent cirkel rond center met kleur voor debug
     */
    void detectDiscs(const Mat& warped, Board& board);

    /**
     * updateBoard:
     * - slaat de nieuw berekende board op
     * - reset timer voor tick-API
     */
    void updateBoard(const Board& board);

    /**
     * tick:
     * - leest nieuw frame
     * - doet detectAndWarpBoard + detectDiscs
     * - vergelijkt met oldBoard_, wacht 3 seconden
     *   totdat board echt stabiel is
     * returnt true als er een stabiele update klaarstaat
     */
    bool tick();

    /** getState: geeft laatste stabiele board terug */
    Board getState() const;

    /** Vergelijkt twee boards, returnt true bij verschil */
    static bool boardsDiffer(const Board& a, const Board& b);

    /** Print comfortabel de board-matrix naar console */
    static void printBoard(const Board& board);

    /** Opent 1 resizable venster met sliders voor alle HSV-parameters */
    void createSliderWindow();

private:
    /** classifyColor: geeft 1=blauw, 2=rood, 0=leeg */
    int classifyColor(const Mat& roiHSV);

    VideoCapture            cap_;         // webcam
    Mat                     perspectiveM_;
    Size                    warpSize_{ 700,600 };

    // interne state voor tick()
    Board                   oldBoard_{}, newBoard_{};
    steady_clock::time_point changeStart_;
    bool                    firstTick_{ true };

    // HSV-ranges in te stellen via sliders
    int hLowRed1_, hHighRed1_, hLowRed2_, hHighRed2_, sLowRed_, vLowRed_;
    int hLowBlue_, hHighBlue_, sLowBlue_, vLowBlue_;
    int hLowBoard_, hHighBoard_, sLowBoard_, vLowBoard_;
};

#endif // GAMEVISION_H
