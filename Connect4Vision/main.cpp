// main.cpp

#include "GameVision.h"
#include <opencv2/opencv.hpp>   // Nodig voor VideoCapture, imshow, waitKey, etc.

using namespace cv;
using namespace std;

int main() {
    // Kies de camera-index (0 = laptop camera, 1 = usb webcam)
    int cameraIndex = 1;
    VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        cerr << "Fout: kon camera niet openen!\n";
        return -1;
    }

    GameVision gv;
    Board board{}, prevBoard{};
    bool firstFrame = true;

    Mat frame, warped;
    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        // Vind en warp het bord
        bool found = gv.detectAndWarpBoard(frame, warped);
        if (found) {
            // Detecteer fiches in het gewarpte bord
            gv.detectDiscs(warped, board);

            if (firstFrame || GameVision::boardsDiffer(board, prevBoard)) {
                GameVision::printBoard(board);
                prevBoard = board;
                firstFrame = false;
            }
            imshow("Gewarpte Connect4", warped);
        }
        else {
            imshow("Gewarpte Connect4", frame);
        }

        // Toon het originele frame altijd
        imshow("Connect4", frame);

        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
