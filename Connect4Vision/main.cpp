#include "GameVision.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// Standalone demo: opent camera, toont sliders en herkent bord + fiches in een blokkerende loop
int main() {
    //USB pakken (index 1)
    VideoCapture camera(1);
    if (!camera.isOpened()) {
        cerr << "Camera niet bereikbaar\n";
        return -1;
    }

    GameVision gv;
    gv.createSliderWindow();  // sliders voor live HSV-
    Board prevBoard{}, currBoard{};
    Mat frame, warped;

    // hoofdloop, breekt op 'q'
    while (waitKey(1) != 'q') {
        // frame lezen
        if (!camera.read(frame))
            continue;  // skip als er niks binnenkomt

        warped = frame.clone();  // init warped met origineel

        // 1) bord detectie + warp
        if (gv.detectAndWarpBoard(frame, warped)) {
            // 2) fiche-detectie in gewarpte image
            gv.detectDiscs(warped, currBoard);

            // 3) alleen printen als matrix echt verandert
            if (GameVision::boardsDiffer(currBoard, prevBoard)) {
                GameVision::printBoard(currBoard);
                prevBoard = currBoard;
            }
        }

        // vensters bijwerken
        imshow("Vervormd", warped);
        imshow("Origineel", frame);
    }
    return 0;
}
