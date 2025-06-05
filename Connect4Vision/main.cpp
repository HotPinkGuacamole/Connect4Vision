// main.cpp

#include "GameVision.h"

int main() {
    // Kies de camera-index (0 = eerste camera, 1 = tweede, etc.)
    int cameraIndex = 1;
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Fout: kon camera met index " << cameraIndex << " niet openen!\n";
        return -1;
    }

    GameVision gv;
    Board board{}, prevBoard{};
    bool firstFrame = true;

    cv::Mat frame, warped;
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

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
