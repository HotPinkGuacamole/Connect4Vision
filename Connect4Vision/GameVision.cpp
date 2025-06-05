// GameVision.cpp

#include "GameVision.h"

using namespace cv;
using namespace std;

GameVision::GameVision()
    :
    // **** ROOD ****
    hLowRed1_(170), hHighRed1_(180),
    hLowRed2_(0), hHighRed2_(10),
    sLowRed_(140), sHighRed_(255),
    vLowRed_(80), vHighRed_(255),

    // **** GROEN ****
    hLowGreen_(35), hHighGreen_(60),
    sLowGreen_(33), sHighGreen_(255),
    vLowGreen_(80), vHighGreen_(255),

    // **** BORD (goud/oranje) ****
    hLowBoard_(10), hHighBoard_(45),
    sLowBoard_(145), sHighBoard_(255),
    vLowBoard_(90), vHighBoard_(255),

    warpSize_(700, 600)
{
    initDefaultHSV();
    createHSVTrackbars();
}

void GameVision::initDefaultHSV() {
    // Beginwaarden zijn al toegewezen via initializer-list
}

void GameVision::createHSVTrackbars() {
    const String winName = "HSV Kalibratie";
    namedWindow(winName, WINDOW_AUTOSIZE);

    // ----- BORD (goud/oranje) -----
    createTrackbar("hLowBoard", winName, &hLowBoard_, 179);
    createTrackbar("hHighBoard", winName, &hHighBoard_, 179);
    createTrackbar("sLowBoard", winName, &sLowBoard_, 255);
    createTrackbar("sHighBoard", winName, &sHighBoard_, 255);
    createTrackbar("vLowBoard", winName, &vLowBoard_, 255);
    createTrackbar("vHighBoard", winName, &vHighBoard_, 255);

    // ----- ROOD (2 hue-ranges) -----
    createTrackbar("hLowRed1", winName, &hLowRed1_, 179);
    createTrackbar("hHighRed1", winName, &hHighRed1_, 179);
    createTrackbar("hLowRed2", winName, &hLowRed2_, 179);
    createTrackbar("hHighRed2", winName, &hHighRed2_, 179);
    createTrackbar("sLowRed", winName, &sLowRed_, 255);
    createTrackbar("sHighRed", winName, &sHighRed_, 255);
    createTrackbar("vLowRed", winName, &vLowRed_, 255);
    createTrackbar("vHighRed", winName, &vHighRed_, 255);

    // ----- GROEN -----
    createTrackbar("hLowGreen", winName, &hLowGreen_, 179);
    createTrackbar("hHighGreen", winName, &hHighGreen_, 179);
    createTrackbar("sLowGreen", winName, &sLowGreen_, 255);
    createTrackbar("sHighGreen", winName, &sHighGreen_, 255);
    createTrackbar("vLowGreen", winName, &vLowGreen_, 255);
    createTrackbar("vHighGreen", winName, &vHighGreen_, 255);
}

void GameVision::calibrateHSV() {
    Mat dummy(1, 1, CV_8UC3, Scalar(0, 0, 0));
    while (true) {
        imshow("HSV Kalibratie", dummy);
        if (waitKey(30) == 'q') {
            destroyWindow("HSV Kalibratie");
            break;
        }
    }
}

bool GameVision::detectAndWarpBoard(const Mat& frame, Mat& warped) {
    Mat hsv, hsvBlur;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    GaussianBlur(hsv, hsvBlur, Size(5, 5), 2);

    // 1. Kleursegmentatie voor het bord (goud/oranje)
    Mat mask;
    inRange(hsvBlur,
        Scalar(hLowBoard_, sLowBoard_, vLowBoard_),
        Scalar(hHighBoard_, sHighBoard_, vHighBoard_),
        mask);

    // 2. Morfologische bewerkingen om ruis/gaten te dichten
    Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1, -1), 3);
    morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1, -1), 1);

    // Toon mask voor kalibratie
    imshow("Board Mask", mask);

    // 3. Contouren zoeken
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;

    // 4. Sorteer contouren op grootte (aflopend)
    sort(contours.begin(), contours.end(),
        [](const vector<Point>& a, const vector<Point>& b) {
            return contourArea(a) > contourArea(b);
        });

    // 5. Loop over contouren om vierhoek te vinden
    for (const auto& c : contours) {
        double area = contourArea(c);
        if (area < 12000) continue;

        vector<Point> approx;
        approxPolyDP(c, approx, arcLength(c, true) * 0.015, true);
        if (approx.size() == 4) {
            // Sorteer en ordeneer hoeken
            sort(approx.begin(), approx.end(),
                [](const Point& p1, const Point& p2) {
                    return (p1.y < p2.y) || (p1.y == p2.y && p1.x < p2.x);
                });
            vector<Point2f> corners(4);
            // Bovenste twee
            if (approx[0].x < approx[1].x) {
                corners[0] = approx[0];
                corners[1] = approx[1];
            }
            else {
                corners[0] = approx[1];
                corners[1] = approx[0];
            }
            // Onderste twee
            if (approx[2].x < approx[3].x) {
                corners[2] = approx[2];
                corners[3] = approx[3];
            }
            else {
                corners[2] = approx[3];
                corners[3] = approx[2];
            }

            // 6. Bereken perspective transform naar 700×600 rechthoek
            vector<Point2f> dstCorners = {
                Point2f(0,   0),
                Point2f(699, 0),
                Point2f(0,   599),
                Point2f(699, 599)
            };
            perspectiveM_ = getPerspectiveTransform(corners, dstCorners);
            warpPerspective(frame, warped, perspectiveM_, warpSize_);
            return true;
        }
    }
    return false;
}

void GameVision::detectDiscs(const Mat& warped, Board& board) {
    Mat hsvWarp;
    cvtColor(warped, hsvWarp, COLOR_BGR2HSV);

    int cellW = warpSize_.width / COLS;
    int cellH = warpSize_.height / ROWS;
    for (auto& row : board) row.fill(0);

    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            int centerX = j * cellW + cellW / 2;
            int centerY = i * cellH + cellH / 2;
            int radius = static_cast<int>(min(cellW, cellH) * 0.4);
            Rect roiRect(centerX - radius, centerY - radius, radius * 2, radius * 2);
            if (roiRect.x < 0 || roiRect.y < 0 ||
                roiRect.x + roiRect.width > warped.cols ||
                roiRect.y + roiRect.height > warped.rows) {
                continue;
            }
            Mat cellHSV = hsvWarp(roiRect);
            int value = classifyColor(cellHSV);
            board[i][j] = value;

            Scalar drawColor(200, 200, 200);
            if (value == 1) drawColor = Scalar(0, 255, 0);
            else if (value == 2) drawColor = Scalar(0, 0, 255);
            circle(warped, Point(centerX, centerY), radius, drawColor, 2);
        }
    }
}

int GameVision::classifyColor(const Mat& roiHSV) {
    Mat maskRed1, maskRed2, maskRed, maskGreen;
    // --- ROOD (2 hue-ranges) ---
    inRange(roiHSV,
        Scalar(hLowRed1_, sLowRed_, vLowRed_),
        Scalar(hHighRed1_, sHighRed_, vHighRed_),
        maskRed1);
    inRange(roiHSV,
        Scalar(hLowRed2_, sLowRed_, vLowRed_),
        Scalar(hHighRed2_, sHighRed_, vHighRed_),
        maskRed2);
    bitwise_or(maskRed1, maskRed2, maskRed);

    // --- GROEN ---
    inRange(roiHSV,
        Scalar(hLowGreen_, sLowGreen_, vLowGreen_),
        Scalar(hHighGreen_, sHighGreen_, vHighGreen_),
        maskGreen);

    double totalPixels = roiHSV.rows * roiHSV.cols;
    double countRed = countNonZero(maskRed);
    double countGreen = countNonZero(maskGreen);

    if (countRed / totalPixels > 0.2)  return 2; // rood/roze → waarde 2
    if (countGreen / totalPixels > 0.2) return 1; // groen → waarde 1
    return 0; // leeg
}

bool GameVision::boardsDiffer(const Board& a, const Board& b) {
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS; ++j)
            if (a[i][j] != b[i][j])
                return true;
    return false;
}

void GameVision::printBoard(const Board& board) {
    cout << "Bordstatus:\n";
    for (const auto& row : board) {
        for (int c : row) cout << c << ' ';
        cout << '\n';
    }
    cout << endl;
}

void GameVision::showDebugWindows(const Mat& warped, const Mat& maskRed, const Mat& maskColor) {
    imshow("Warped Board", warped);
    imshow("Mask Rood", maskRed);
    imshow("Mask Color", maskColor);
}
