#include "GameVision.h"

using namespace cv;
using namespace std;

GameVision::GameVision()
    :
    // ROOD/ROZE
    hLowRed1_(170), hHighRed1_(180),
    hLowRed2_(0), hHighRed2_(10),
    sLowRed_(140), vLowRed_(80),

    // GROEN
    hLowGreen_(64), hHighGreen_(82),
    sLowGreen_(50), vLowGreen_(108),

    // BORD (goud/oranje)
    hLowBoard_(16), hHighBoard_(24),
    sLowBoard_(86), vLowBoard_(35),

    warpSize_(700, 600)
{
    initDefaultHSV();
    createHSVTrackbars();
}

void GameVision::initDefaultHSV() {
    // default values set in initializer
}

void GameVision::createHSVTrackbars() {
    const String winName = "HSV Kalibratie";
    namedWindow(winName, WINDOW_NORMAL);

    // BORD (goud/oranje)
    createTrackbar("Board Hue Low", winName, &hLowBoard_, 179);
    createTrackbar("Board Hue High", winName, &hHighBoard_, 179);
    createTrackbar("Board Sat Low", winName, &sLowBoard_, 255);
    createTrackbar("Board Val Low", winName, &vLowBoard_, 255);

    // ROOD/ROZE
    createTrackbar("Red Hue1 Low", winName, &hLowRed1_, 179);
    createTrackbar("Red Hue1 High", winName, &hHighRed1_, 179);
    createTrackbar("Red Hue2 Low", winName, &hLowRed2_, 179);
    createTrackbar("Red Hue2 High", winName, &hHighRed2_, 179);
    createTrackbar("Red Sat Low", winName, &sLowRed_, 255);
    createTrackbar("Red Val Low", winName, &vLowRed_, 255);

    // GROEN
    createTrackbar("Green Hue Low", winName, &hLowGreen_, 179);
    createTrackbar("Green Hue High", winName, &hHighGreen_, 179);
    createTrackbar("Green Sat Low", winName, &sLowGreen_, 255);
    createTrackbar("Green Val Low", winName, &vLowGreen_, 255);
}

bool GameVision::detectAndWarpBoard(const cv::Mat& frame, cv::Mat& warped) {
    // 1. Converteer naar HSV en vervaag om ruis te verminderen
    cv::Mat hsv, blurred;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::GaussianBlur(hsv, blurred, cv::Size(5, 5), 2);

    // 2. Maak een masker voor de goud/oranje bordkleur
    cv::Mat mask;
    cv::inRange(
        blurred,
        cv::Scalar(hLowBoard_, sLowBoard_, vLowBoard_),
        cv::Scalar(hHighBoard_, 255, 255),
        mask
    );

    // 3. Morfologische bewerkingen: dicht kleine gaten en verwijder ruis
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);

    // Optioneel debugvenster
    cv::imshow("Board Mask", mask);

    // 4. Zoek alle externe contouren in het masker
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return false;
    }

    // 5. Sorteer contouren op grootte (grootste eerst)
    std::sort(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) > cv::contourArea(b);
        }
    );

    // 6. Probeer de grootste contour tot een vierhoek te benaderen
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 12000) {
            continue;  // negeer te kleine contouren
        }

        // Benader contour met polygon
        std::vector<cv::Point> approx;
        cv::approxPolyDP(
            contour,
            approx,
            cv::arcLength(contour, true) * 0.015,
            true
        );

        // Alleen verder als het een vierhoek is
        if (approx.size() == 4) {
            // 7. Orden de 4 hoekpunten: TL, TR, BL, BR
            std::sort(approx.begin(), approx.end(),
                [](const cv::Point& p1, const cv::Point& p2) {
                    return (p1.y < p2.y) || (p1.y == p2.y && p1.x < p2.x);
                }
            );
            std::vector<cv::Point2f> corners(4);
            // Top twee
            if (approx[0].x < approx[1].x) {
                corners[0] = approx[0];
                corners[1] = approx[1];
            }
            else {
                corners[0] = approx[1];
                corners[1] = approx[0];
            }
            // Bottom twee
            if (approx[2].x < approx[3].x) {
                corners[2] = approx[2];
                corners[3] = approx[3];
            }
            else {
                corners[2] = approx[3];
                corners[3] = approx[2];
            }

            // 8. Bereken de homografie en warp het frame
            std::vector<cv::Point2f> dstCorners = {
                {0.0f,   0.0f},
                {699.0f, 0.0f},
                {0.0f,   599.0f},
                {699.0f, 599.0f}
            };
            perspectiveM_ = cv::getPerspectiveTransform(corners, dstCorners);
            cv::warpPerspective(frame, warped, perspectiveM_, warpSize_);

            return true;
        }
    }

    // Geen geldige vierhoek gevonden
    return false;
}


void GameVision::detectDiscs(const Mat& warped, Board& board) {
    Mat hsv;
    cvtColor(warped, hsv, COLOR_BGR2HSV);

    int cellW = warpSize_.width / COLS;
    int cellH = warpSize_.height / ROWS;
    for (auto& r : board) r.fill(0);

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int cx = j * cellW + cellW / 2, cy = i * cellH + cellH / 2;
            int rad = min(cellW, cellH) / 3;
            Rect roi(cx - rad, cy - rad, 2 * rad, 2 * rad);
            if (roi.x<0 || roi.y<0 || roi.x + roi.width>hsv.cols || roi.y + roi.height>hsv.rows)
                continue;
            Mat cell = hsv(roi);
            int v = classifyColor(cell);
            board[i][j] = v;
            Scalar col = (v == 1 ? Scalar(0, 255, 0) : (v == 2 ? Scalar(0, 0, 255) : Scalar(200, 200, 200)));
            circle((Mat&)warped, Point(cx, cy), rad, col, 2);
        }
    }
}

int GameVision::classifyColor(const cv::Mat& roiHSV) {
    // 1. Maak maskers voor de twee rode hue-ranges
    cv::Mat maskRed1, maskRed2, maskRed;
    cv::inRange(
        roiHSV,
        cv::Scalar(hLowRed1_, sLowRed_, vLowRed_),
        cv::Scalar(hHighRed1_, 255, 255),
        maskRed1
    );
    cv::inRange(
        roiHSV,
        cv::Scalar(hLowRed2_, sLowRed_, vLowRed_),
        cv::Scalar(hHighRed2_, 255, 255),
        maskRed2
    );
    // Combineer de twee rode maskers
    cv::bitwise_or(maskRed1, maskRed2, maskRed);

    // 2. Maak masker voor groen
    cv::Mat maskGreen;
    cv::inRange(
        roiHSV,
        cv::Scalar(hLowGreen_, sLowGreen_, vLowGreen_),
        cv::Scalar(hHighGreen_, 255, 255),
        maskGreen
    );

    // 3. Tel het aantal gemaskeerde pixels
    double totalPixels = static_cast<double>(roiHSV.rows * roiHSV.cols);
    double countRed = static_cast<double>(cv::countNonZero(maskRed));
    double countGreen = static_cast<double>(cv::countNonZero(maskGreen));

    // 4. Classificeer op basis van >20% drempel
    if ((countGreen / totalPixels) > 0.2) {
        return 1;  // voornamelijk groen
    }
    if ((countRed / totalPixels) > 0.2) {
        return 2;  // voornamelijk rood/roze
    }

    return 0;      // leeg of geen dominante kleur
}

bool GameVision::boardsDiffer(const Board& a, const Board& b) {
    // Loop over alle rijen
    for (int row = 0; row < ROWS; ++row) {
        // Loop over alle kolommen in deze rij
        for (int col = 0; col < COLS; ++col) {
            // Als de waarden niet gelijk zijn: borden verschillen
            if (a[row][col] != b[row][col]) {
                return true;
            }
        }
    }
    // Geen verschillen gevonden: borden zijn gelijk
    return false;
}

void GameVision::printBoard(const Board& board) {
    // Geef koptekst weer
    std::cout << "Bordstatus:\n";

    // Loop elke rij af
    for (const auto& row : board) {
        // Print alle cellen in de rij
        for (int cell : row) {
            std::cout << cell << ' ';
        }
        // Nieuwe regel na de rij
        std::cout << '\n';
    }

    // Extra lege regel en flush
    std::cout << std::endl;
}

