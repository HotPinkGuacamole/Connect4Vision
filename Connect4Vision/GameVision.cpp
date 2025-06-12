#include "GameVision.h"
using namespace cv;
using namespace std;
using namespace std::chrono;

/**
 * Constructor:
 * - opent camera op gegeven index
 * - initialiseert lege boards + timestamp
 * - zet tunede HSV-waardes
 */
GameVision::GameVision(int cameraIndex)
    : cap_(cameraIndex),
    warpSize_{ 700,600 }
{
    if (!cap_.isOpened())
        throw runtime_error("GameVision: kon webcam niet openen");

    // init lege boards
    for (auto& r : oldBoard_) r.fill(0);
    newBoard_ = oldBoard_;
    changeStart_ = steady_clock::now();

    // tunede HSV‐waarden
    hLowRed1_ = 147;  hHighRed1_ = 179;
    hLowRed2_ = 0;  hHighRed2_ = 10;
    sLowRed_ = 96;  vLowRed_ = 80;
    hLowBlue_ = 103;  hHighBlue_ = 113;
    sLowBlue_ = 98;  vLowBlue_ = 137;
    hLowBoard_ = 0;  hHighBoard_ = 117;
    sLowBoard_ = 0;  vLowBoard_ = 129;
}

/** Destructor: vernietig alle OpenCV-vensters */
GameVision::~GameVision() noexcept {
    destroyAllWindows();
}

/**
 * createSliderWindow:
 * maakt 1 window met sliders (trackbars) voor alle HSV-ranges
 */
void GameVision::createSliderWindow() {
    const char* win = "HSV Kalibratie";
    namedWindow(win, WINDOW_NORMAL);

    // ---- bord (goud/oranje) ----
    createTrackbar("Bord Hue Laag", win, &hLowBoard_, 179);
    createTrackbar("Bord Hue Hoog", win, &hHighBoard_, 179);
    createTrackbar("Bord Sat Laag", win, &sLowBoard_, 255);
    createTrackbar("Bord Val Laag", win, &vLowBoard_, 255);

    // ---- rood/roze ----
    createTrackbar("Rood Hue1 Laag", win, &hLowRed1_, 179);
    createTrackbar("Rood Hue1 Hoog", win, &hHighRed1_, 179);
    createTrackbar("Rood Hue2 Laag", win, &hLowRed2_, 179);
    createTrackbar("Rood Hue2 Hoog", win, &hHighRed2_, 179);
    createTrackbar("Rood Sat Laag", win, &sLowRed_, 255);
    createTrackbar("Rood Val Laag", win, &vLowRed_, 255);

    // ---- blauw ----
    createTrackbar("Blauw Hue Laag", win, &hLowBlue_, 179);
    createTrackbar("Blauw Hue Hoog", win, &hHighBlue_, 179);
    createTrackbar("Blauw Sat Laag", win, &sLowBlue_, 255);
    createTrackbar("Blauw Val Laag", win, &vLowBlue_, 255);
}

/**
 * detectAndWarpBoard:
 * 1) converteer naar HSV + blur
 * 2) inRange op bordkleur
 * 3) morph close/open om gaten te vullen
 * 4) findContours, sorteer, approxPolyDP
 * 5) als vierhoek, getPerspectiveTransform + warp
 */
bool GameVision::detectAndWarpBoard(const Mat& frame, Mat& warped) {
    Mat hsv, blurHSV, mask;
    cvtColor(frame, hsv, COLOR_BGR2HSV);                 // BGR→HSV
    GaussianBlur(hsv, blurHSV, Size(5, 5), 2);            // beetje blur

    // kleurfilter bord
    inRange(blurHSV,
        Scalar(hLowBoard_, sLowBoard_, vLowBoard_),
        Scalar(hHighBoard_, 255, 255),
        mask);

    // morfologie om ruis te dichten
    static const Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1, -1), 3);
    morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1, -1), 1);

    imshow("Board Mask", mask);  // debugvenster

    // contouren zoeken
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;

    // grootste eerst
    sort(contours.begin(), contours.end(),
        [](auto& a, auto& b) { return contourArea(a) > contourArea(b); });

    for (auto& c : contours) {
        if (contourArea(c) < 12000) continue;  // te klein negeren

        // benader met polygon
        vector<Point> approx;
        approxPolyDP(c, approx, arcLength(c, true) * 0.015, true);
        if (approx.size() != 4) continue;

        // hoekpunten sorteren op y, dan x
        sort(approx.begin(), approx.end(),
            [](auto& p1, auto& p2) {
                return (p1.y < p2.y) || (p1.y == p2.y && p1.x < p2.x);
            });

        // top en bottom regels
        vector<Point2f> corners(4), dst{ {0,0},{699,0},{0,599},{699,599} };
        if (approx[0].x < approx[1].x) {
            corners[0] = approx[0]; corners[1] = approx[1];
        }
        else {
            corners[0] = approx[1]; corners[1] = approx[0];
        }
        if (approx[2].x < approx[3].x) {
            corners[2] = approx[2]; corners[3] = approx[3];
        }
        else {
            corners[2] = approx[3]; corners[3] = approx[2];
        }

        // warp
        perspectiveM_ = getPerspectiveTransform(corners, dst);
        warpPerspective(frame, warped, perspectiveM_, warpSize_);
        return true;
    }
    return false;
}

/**
 * detectDiscs:
 * - verdeel het gewarpte beeld in 6×7 cellen
 * - voor elke cel een kleine HSV-ROI
 * - klasificeer met classifyColor()
 * - teken cirkel in debug
 */
void GameVision::detectDiscs(const Mat& warped, Board& board) {
    Mat hsv;
    cvtColor(warped, hsv, COLOR_BGR2HSV);

    int cw = warpSize_.width / COLS;
    int ch = warpSize_.height / ROWS;
    for (auto& row : board) row.fill(0);  // reset board

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int cx = j * cw + cw / 2, cy = i * ch + ch / 2;
            int r = min(cw, ch) / 3;
            Rect roi(cx - r, cy - r, 2 * r, 2 * r);
            if (roi.x<0 || roi.y<0 || roi.br().x>hsv.cols || roi.br().y>hsv.rows)
                continue;
            Mat cell = hsv(roi);  // kleine ROI

            int val = classifyColor(cell);  // 1=blauw,2=rood,0=leeg
            board[i][j] = val;

            // debug: cirkel tekenen
            Scalar col = (val == 1 ? Scalar(255, 0, 0) :  // blauw
                val == 2 ? Scalar(0, 0, 255) :  // rood
                Scalar(200, 200, 200)); // leeg
            circle(warped, Point(cx, cy), r, col, 2);
        }
    }
}

/**
 * classifyColor:
 * - bouwt maskers voor rood (2 hue-ranges) en blauw
 * - telt non-zero pixels
 * - return 1 bij blauw >20%, 2 bij rood >20%, anders 0
 */
int GameVision::classifyColor(const Mat& cell) {
    Mat r1, r2, mR, mB;
    // rood maskers
    inRange(cell, Scalar(hLowRed1_, sLowRed_, vLowRed_), Scalar(hHighRed1_, 255, 255), r1);
    inRange(cell, Scalar(hLowRed2_, sLowRed_, vLowRed_), Scalar(hHighRed2_, 255, 255), r2);
    bitwise_or(r1, r2, mR);
    // blauw masker
    inRange(cell, Scalar(hLowBlue_, sLowBlue_, vLowBlue_), Scalar(hHighBlue_, 255, 255), mB);

    double tot = double(cell.rows * cell.cols);
    if (countNonZero(mB) / tot > 0.2) return 1;
    if (countNonZero(mR) / tot > 0.2) return 2;
    return 0;
}

/** Vergelijkt twee boards; true bij verschil */
bool GameVision::boardsDiffer(const Board& a, const Board& b) {
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            if (a[i][j] != b[i][j]) return true;
    return false;
}

/** Print board-matrix netjes in console */
void GameVision::printBoard(const Board& board) {
    cout << "Bordstatus:\n";
    for (auto& row : board) {
        for (int c : row) cout << c << ' ';
        cout << "\n";
    }
    cout << flush;
}

/** updateBoard: reset old/new board en timestamp voor tick() */
void GameVision::updateBoard(const Board& board) {
    oldBoard_ = newBoard_;
    newBoard_ = board;
    changeStart_ = steady_clock::now();
}

/**
 * tick:
 * - blokkeert niet
 * - checkt nieuw board, wacht totdat het 3s stabiel is
 * - return true als stabiele nieuwe board klaarstaat
 */
bool GameVision::tick() {
    Mat frame, warped;
    cap_ >> frame;
    if (frame.empty()) {
        changeStart_ = steady_clock::now();
        return false;
    }
    if (!detectAndWarpBoard(frame, warped)) {
        changeStart_ = steady_clock::now();
        return false;
    }
    detectDiscs(warped, newBoard_);
    if (boardsDiffer(newBoard_, oldBoard_)) {
        if (firstTick_) {
            firstTick_ = false;
            changeStart_ = steady_clock::now();
            return false;
        }
        if (steady_clock::now() - changeStart_ > seconds(3)) {
            oldBoard_ = newBoard_;
            return true;
        }
    }
    else {
        changeStart_ = steady_clock::now();
    }
    return false;
}

/** getState: retourneert laatste stabiele board */
Board GameVision::getState() const {
    return newBoard_;
}
