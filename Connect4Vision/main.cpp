#include "GameVision.h"

int main() {
    VideoCapture camera(1);
    if (!camera.isOpened()) {
        cerr << "Camera niet bereikbaar\n";
        return -1;
    }

    GameVision gv;
    gv.createSliderWindow();

    Board prevBoard{}, currBoard{};
    Mat frame, warped;

    while (waitKey(1) != 'q') {
        if (!camera.read(frame))
            continue;

        warped = frame.clone();

        if (gv.detectAndWarpBoard(frame, warped)) {
            gv.detectDiscs(warped, currBoard);
            if (GameVision::boardsDiffer(currBoard, prevBoard)) {
                GameVision::printBoard(currBoard);
                prevBoard = currBoard;
            }
        }

        imshow("Vervormd", warped);
        imshow("Origineel", frame);
    }
    return 0;
}
