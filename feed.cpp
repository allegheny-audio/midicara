#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <math.h>
#include <chrono>
#include <ctime>
#include <memory>
#include <libremidi/libremidi.hpp>
#include "ftxui/component/captured_mouse.hpp"
#include "ftxui/component/component.hpp"
#include "ftxui/component/component_base.hpp"
#include "ftxui/component/component_options.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/node.hpp"
#include "ftxui/screen/color.hpp"
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/canvas.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/color.hpp>

using namespace dlib;
using namespace std;

#if DEBUG
static const bool isDebug = true;
#else
static const bool isDebug = false;
#endif

std::mutex m;
std::mutex midiMutex;
std::mutex tuiMutex;
std::atomic<bool> run = true;

ftxui::ScreenInteractive screen = ftxui::ScreenInteractive::Fullscreen();
const unsigned int capPropFrameWidth = 100;
const unsigned int capPropFrameHeight = 100;
unsigned int cvMatrixWidth = 100;
unsigned int cvMatrixHeight = 100;

bool calibrationGreenLight = false;
bool noseZeroCalibrationComplete = false;
bool mouthCalibrationClosedComplete = false;
bool mouthCalibrationOpenUpDownComplete = false;
bool mouthCalibrationOpenRightLeftComplete = false;
bool nosePitchBoardUpperLeftComplete = false;
bool nosePitchBoardUpperRightComplete = false;
bool nosePitchBoardLowerRightComplete = false;
bool nosePitchBoardLowerLeftComplete = false;
bool nosePitchBoardCalculateComplete = false;

int dlibFacialParts[68][2];
std::atomic<bool> faceDetected = false;
unsigned int noseSensitivity = 2; // sensitivity multiplier to nose pointer position (because nose is fairly accurate, it can be increased in sensitivity)
int noseCurrentPosition[2];
int noseCurrentPositionMultiplied[2]; // current nose position multiplied by the sensitivity factor
int noseZeroCalibrated[2];
int mouthOuterLipUpDownCurrentDistance;
int mouthOuterLipRightLeftCurrentDistance;
int mouthInnerLipUpDownCurrentDistance;
int mouthInnerLipRightLeftCurrentDistance;
int mouthOuterLipUpDownClosedDistanceCalibrated;
int mouthOuterLipRightLeftClosedDistanceCalibrated;
int mouthInnerLipUpDownClosedDistanceCalibrated;
int mouthInnerLipRightLeftClosedDistanceCalibrated;
int mouthOuterLipUpDownOpenDistanceCalibrated;
int mouthInnerLipUpDownOpenDistanceCalibrated;
int mouthOuterLipRightLeftOpenDistanceCalibrated;
int mouthInnerLipRightLeftOpenDistanceCalibrated;
int nosePitchBoardUpperLeftPosition[2];
int nosePitchBoardUpperRightPosition[2];
int nosePitchBoardLowerRightPosition[2];
int nosePitchBoardLowerLeftPosition[2];
unsigned int pitchBoardNumberOfStrings = 4;
unsigned int pitchBoardNumberOfFrets;
float*** pitchBoardPointsAsFloats;
unsigned int*** pitchBoardPoints;
unsigned char** pixelMidiNoteCache;

std::atomic<bool> midiPortSelected = false;
std::atomic<bool> midiPortsPopulated = false;
std::vector<std::string> midiPortMenuEntries;
int midiPortMenuSelectedIndex = std::numeric_limits<int>::max();
bool midiPlaying;
bool mouthOpen;
float mouthOpenPercentage;
float mouthWidePercentage;
unsigned int midiNoteToPlay;
unsigned int noseDistanceMagnitude;

unsigned int polyphonyCacheSize = 6;
struct polyphonyCacheItem {
  unsigned int note;
  std::chrono::high_resolution_clock::time_point entryTime;
};
polyphonyCacheItem* polyphonyCache;

// for debugging purposes: show an opencv matrix on a display window
void opencvShowGrayscaleMatrix(dlib::image_window* win, cv::Mat* mat) {
  cv_image<unsigned char> myImage(*mat);
  win->clear_overlay();
  win->set_image(myImage);
}

// create a 2D parameterized line function that starts from p1, f, and return f(t)
float* parametricLine(float p1[], float p2[], float t) {
  float* res = (float*)malloc(sizeof(float) * 2);
  res[0] = p1[0] + (p2[0] - p1[0]) * t;
  res[1] = p1[1] + (p2[1] - p1[1]) * t;
  return res;
}
float* parametricLine(int p1[], int p2[], float t) {
  float p1_f[2], p2_f[2];
  p1_f[0] = (float)p1[0];
  p1_f[1] = (float)p1[1];
  p2_f[0] = (float)p2[0];
  p2_f[1] = (float)p2[1];
  return parametricLine((float*)p1_f, (float*)p2_f, t);
}
float triangleArea(int x1, int y1, int x2, int y2, int x3, int y3) {
  return (float)std::abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0f;
}
float triangleArea(unsigned int p1[], unsigned int p2[], unsigned int p3[]) {
  return triangleArea((int)p1[0], (int)p1[1], (int)p2[0], (int)p2[1], (int)p3[0], p3[1]);
}

void interpretFacialData() {
  // wait for calibration before running
  while (run.load()) {
    m.lock();
    if (true
      && noseZeroCalibrationComplete
      && nosePitchBoardCalculateComplete
      && mouthCalibrationClosedComplete
      && mouthCalibrationOpenUpDownComplete
      && mouthCalibrationOpenRightLeftComplete) {
      m.unlock();
      break;
    }
    m.unlock();
    std::this_thread::sleep_for(100ms);
  }
  auto percentageNormalize = [](float percentage) -> float {
    if (percentage > 1.0) {
      return 1.0;
    } else if (percentage < 0) {
      return 0;
    } else {
      return percentage;
    }
  };
  while (run.load()) {
    midiMutex.lock();
    midiNoteToPlay = pixelMidiNoteCache[noseCurrentPositionMultiplied[0]][noseCurrentPositionMultiplied[1]];
    midiMutex.unlock();

    // handle mouth dimensions
    m.lock();
    mouthWidePercentage = percentageNormalize((float)(
        (float)((mouthOuterLipRightLeftCurrentDistance - mouthOuterLipRightLeftClosedDistanceCalibrated + mouthInnerLipRightLeftCurrentDistance - mouthInnerLipRightLeftClosedDistanceCalibrated) / 2.0)
        /
        (float)((mouthOuterLipRightLeftOpenDistanceCalibrated - mouthOuterLipRightLeftClosedDistanceCalibrated + mouthInnerLipRightLeftOpenDistanceCalibrated - mouthInnerLipRightLeftClosedDistanceCalibrated) / 2.0)
    ));
    if (mouthOuterLipUpDownCurrentDistance > mouthOuterLipUpDownClosedDistanceCalibrated) {
      midiMutex.lock();
      mouthOpen = true;
      mouthOpenPercentage = percentageNormalize((float)(
          (float)((mouthOuterLipUpDownCurrentDistance - mouthOuterLipUpDownClosedDistanceCalibrated + mouthInnerLipUpDownCurrentDistance - mouthInnerLipUpDownClosedDistanceCalibrated) / 2.0)
          /
          (float)((mouthOuterLipUpDownOpenDistanceCalibrated - mouthOuterLipUpDownClosedDistanceCalibrated + mouthInnerLipUpDownOpenDistanceCalibrated - mouthInnerLipUpDownClosedDistanceCalibrated) / 2.0)
      ));
      midiMutex.unlock();
    } else {
      midiMutex.lock();
      mouthOpen = false;
      mouthOpenPercentage = 0.0;
      midiMutex.unlock();
    }
    m.unlock();
  }
}

void calculatePitchBoardPoints() {
  // calculate reduced row eschelon form.
  // Stolen from: https://stackoverflow.com/questions/31756413/solving-a-simple-matrix-in-row-reduced-form-in-c
  auto rowReduce = [](float A[2][3]) {
    float res[2][3];
    const int nrows = 2;
    const int ncols = 3;
    int lead = 0; 
    while (lead < nrows) {
      float d, m;
      for (int r = 0; r < nrows; r++) { // for each row ...
        /* calculate divisor and multiplier */
        d = A[lead][lead];
        m = A[r][lead] / A[lead][lead];
        for (int c = 0; c < ncols; c++) { // for each column ...
          if (r == lead)
            A[r][c] /= d;               // make pivot = 1
          else
            A[r][c] -= A[lead][c] * m;  // make other = 0
        }
      }
      lead++;
    }
  };
  auto intersectionOfTwoLinesGivenFourPoints = [rowReduce](float p1[2], float p2[2], float q1[2], float q2[2]) -> float* {
    // create matrix to represent system of equations
    float matrix[2][3] = {
      { (p2[0] - p1[0]) * -1, q2[0] - q1[0], p1[0] - q1[0] },
      { (p2[1] - p1[1]) * -1, q2[1] - q1[1], p1[1] - q1[1] }
    };
    // convert the matrix to reduced row eschelon form
    rowReduce(matrix);
    // check that both solved parameters produce the same point
    // rounding here in order to avoid idiosyncrasies from binary approx
    if (true
      && round(parametricLine(p1, p2, matrix[0][2])[0]) == round(parametricLine(q1, q2, matrix[1][2])[0])
      && round(parametricLine(p1, p2, matrix[0][2])[1]) == round(parametricLine(q1, q2, matrix[1][2])[1])) {
      return parametricLine(p1, p2, matrix[0][2]);
    } else {
      cerr << "WARNING: could not find coincident point of two lines." << endl;
      return p1; // default return value
    }
  };
  m.lock();
  // allocate memory for 2D array of points, hence making a 3D array
  pitchBoardPointsAsFloats = (float***)malloc(sizeof(float**) * (pitchBoardNumberOfFrets + 1));
  pitchBoardPoints = (unsigned int***)malloc(sizeof(unsigned int**) * (pitchBoardNumberOfFrets + 1));
  for (int i = 0; i <= pitchBoardNumberOfFrets; i++) {
    pitchBoardPointsAsFloats[i] = (float**)malloc(sizeof(float*) * (pitchBoardNumberOfStrings + 1));
    pitchBoardPoints[i] = (unsigned int**)malloc(sizeof(unsigned int*) * (pitchBoardNumberOfStrings + 1));
    for (int j = 0; j <= pitchBoardNumberOfStrings; j++) {
      pitchBoardPointsAsFloats[i][j] = (float*)malloc(sizeof(float) * 2);
      pitchBoardPoints[i][j] = (unsigned int*)malloc(sizeof(unsigned int) * 2);
    }
  }
  // calculate points along horizontal boundary lines
  for (int i = 0; i <= pitchBoardNumberOfFrets; i++) {
    // point along top horizontal line
    pitchBoardPointsAsFloats[i][0][0] = parametricLine(nosePitchBoardUpperLeftPosition, nosePitchBoardUpperRightPosition, (float)i / (float)pitchBoardNumberOfFrets)[0];
    pitchBoardPointsAsFloats[i][0][1] = parametricLine(nosePitchBoardUpperLeftPosition, nosePitchBoardUpperRightPosition, (float)i / (float)pitchBoardNumberOfFrets)[1];
    // point along bottom horizontal line
    pitchBoardPointsAsFloats[i][pitchBoardNumberOfStrings][0] = parametricLine(nosePitchBoardLowerLeftPosition, nosePitchBoardLowerRightPosition, (float)i / (float)pitchBoardNumberOfFrets)[0];
    pitchBoardPointsAsFloats[i][pitchBoardNumberOfStrings][1] = parametricLine(nosePitchBoardLowerLeftPosition, nosePitchBoardLowerRightPosition, (float)i / (float)pitchBoardNumberOfFrets)[1];
  }
  // calculate points along vertical boundary lines
  for (int j = 0; j <= pitchBoardNumberOfStrings; j++) {
    // point along left vertical line
    pitchBoardPointsAsFloats[0][j][0] = parametricLine(nosePitchBoardUpperLeftPosition, nosePitchBoardLowerLeftPosition, (float)j / (float)pitchBoardNumberOfStrings)[0];
    pitchBoardPointsAsFloats[0][j][1] = parametricLine(nosePitchBoardUpperLeftPosition, nosePitchBoardLowerLeftPosition, (float)j / (float)pitchBoardNumberOfStrings)[1];
    // point along right vertical line
    pitchBoardPointsAsFloats[pitchBoardNumberOfFrets][j][0] = parametricLine(nosePitchBoardUpperRightPosition, nosePitchBoardLowerRightPosition, (float)j / (float)pitchBoardNumberOfStrings)[0];
    pitchBoardPointsAsFloats[pitchBoardNumberOfFrets][j][1] = parametricLine(nosePitchBoardUpperRightPosition, nosePitchBoardLowerRightPosition, (float)j / (float)pitchBoardNumberOfStrings)[1];
  }

  for (int i = 1; i < pitchBoardNumberOfFrets; i++) {
    for (int j = 1; j < pitchBoardNumberOfStrings; j++) {
      pitchBoardPointsAsFloats[i][j][0] = intersectionOfTwoLinesGivenFourPoints(
        pitchBoardPointsAsFloats[i][0],
        pitchBoardPointsAsFloats[i][pitchBoardNumberOfStrings],
        pitchBoardPointsAsFloats[0][j],
        pitchBoardPointsAsFloats[pitchBoardNumberOfFrets][j]
      )[0];
      pitchBoardPointsAsFloats[i][j][1] = intersectionOfTwoLinesGivenFourPoints(
        pitchBoardPointsAsFloats[i][0],
        pitchBoardPointsAsFloats[i][pitchBoardNumberOfStrings],
        pitchBoardPointsAsFloats[0][j],
        pitchBoardPointsAsFloats[pitchBoardNumberOfFrets][j]
      )[1];
    }
  }
  // convert all points to integers by rounding them, so they can be drawn on a canvas-like object
  for (int i = 0; i <= pitchBoardNumberOfFrets; i++) {
    for (int j = 0; j <= pitchBoardNumberOfStrings; j++) {
      pitchBoardPoints[i][j][0] = (unsigned int)round(pitchBoardPointsAsFloats[i][j][0]);
      pitchBoardPoints[i][j][1] = (unsigned int)round(pitchBoardPointsAsFloats[i][j][1]);
    }
  }
  unsigned char** pitchBoardMidiNoteCache;
  // allocate a cache that relates to each 'note' on the pitch board
  pitchBoardMidiNoteCache = (unsigned char**)malloc(sizeof(unsigned char*) * pitchBoardNumberOfFrets);
  for (int i = 0; i < pitchBoardNumberOfFrets; i++) {
    pitchBoardMidiNoteCache[i] = (unsigned char*)malloc(sizeof(unsigned char) * pitchBoardNumberOfStrings);
  }
  // calculate what the midi note that would be played for each quadrilateral
  int midiCounter = 0;
  unsigned char G3Midi = 55;
  for (int j = pitchBoardNumberOfStrings - 1; j >= 0; j--) {
    for (int i = 0; i < pitchBoardNumberOfFrets; i++) {
      pitchBoardMidiNoteCache[i][j] = G3Midi + midiCounter;
      midiCounter++;
    }
  }
  // store mapping between pixel and midi note to play to avoid calculations on-the-fly
  pixelMidiNoteCache = (unsigned char**)malloc(sizeof(unsigned char*) * cvMatrixWidth);
  for (int i = 0; i < cvMatrixWidth; i++) {
    pixelMidiNoteCache[i] = (unsigned char*)malloc(sizeof(unsigned char) * cvMatrixHeight);
  }
  for (int u = 0; u < cvMatrixWidth; u++) {
    for (int v = 0; v < cvMatrixHeight; v++) {
      // determine and cache which part of the pitchBoardMidiNoteCache this pixel belongs to
      bool boundingBoxFound = false;
      for (int i = 0; i < pitchBoardNumberOfFrets; i++) {
        if (boundingBoxFound)
          break;
        for (int j = 0; j < pitchBoardNumberOfStrings; j++) {
          if (boundingBoxFound)
            break;
          // calculate area by breaking quadrilateral into two triangles
          // first triangle    second triangle
          // *   *             *
          //     *             *   *
          float areaOfCurrentPitchBoardQuadrilateral =
            triangleArea(pitchBoardPoints[i][j], pitchBoardPoints[i + 1][j],pitchBoardPoints[i + 1][j + 1])
            + triangleArea(pitchBoardPoints[i][j], pitchBoardPoints[i][j + 1], pitchBoardPoints[i + 1][j + 1]);
          // quadrilateral area using point in question
          // broken up into four triangles and summed
          float quadrilateralAreaToCompare =
            triangleArea(
                u,
                v,
                (int)pitchBoardPoints[i][j][0],
                (int)pitchBoardPoints[i][j][1],
                (int)pitchBoardPoints[i+1][j][0],
                (int)pitchBoardPoints[i+1][j][1])
            + triangleArea(
                u,
                v,
                (int)pitchBoardPoints[i+1][j][0],
                (int)pitchBoardPoints[i+1][j][1],
                (int)pitchBoardPoints[i+1][j+1][0],
                (int)pitchBoardPoints[i+1][j+1][1])
            + triangleArea(
                u,
                v,
                (int)pitchBoardPoints[i+1][j+1][0],
                (int)pitchBoardPoints[i+1][j+1][1],
                (int)pitchBoardPoints[i][j+1][0],
                (int)pitchBoardPoints[i][j+1][1])
            + triangleArea(
                u,
                v,
                (int)pitchBoardPoints[i][j+1][0],
                (int)pitchBoardPoints[i][j+1][1],
                (int)pitchBoardPoints[i][j][0],
                (int)pitchBoardPoints[i][j][1]);
          if (quadrilateralAreaToCompare == areaOfCurrentPitchBoardQuadrilateral) {
            pixelMidiNoteCache[u][v] = pitchBoardMidiNoteCache[i][j];
            boundingBoxFound = true;
          }
        }
      }
    }
  }
  nosePitchBoardCalculateComplete = true;
  m.unlock();
}

void tuiRenderer() {
  auto calcAverage = [](int* readValue, int* assignValue, int iterations) {
    int measurements[iterations];
    int acc = 0;
    for (int i = 0; i < iterations; i++) {
      m.lock();
      measurements[i] = *readValue;
      m.unlock();
      std::this_thread::sleep_for(3ms);
    }
    for (int i = 0; i < iterations; i++) {
      acc += measurements[i];
    }
    *assignValue = acc / iterations;
  };

  std::vector<std::string> mainMenuEntries = {
    "Midi Setup",
    "Nose Calibration",
    "Mouth Calibration (1/3)",
    "Mouth Calibration (2/3)",
    "Mouth Calibration (3/3)",
    "Pitch Board Calibration (1/5)",
    "Pitch Board Calibration (2/5)",
    "Pitch Board Calibration (3/5)",
    "Pitch Board Calibration (4/5)",
    "Pitch Board Calibration (5/5)",
  };
  int mainMenuSelectedIndex;
  auto mainMenu = ftxui::Menu(&mainMenuEntries, &mainMenuSelectedIndex);

  // wait for midi ports to be populated
  while (!midiPortsPopulated.load()) {
  }
  if (!run.load()) {
    return;
  }
  ftxui::MenuOption midiMenuOption;
  int midiPortMenuSelectedIndexLocal;
  midiMenuOption.on_enter = [&midiPortMenuSelectedIndexLocal] {
    std::unique_lock<std::mutex> lock(midiMutex);
    midiPortMenuSelectedIndex = midiPortMenuSelectedIndexLocal;
    midiPortSelected.store(true);
  };
  auto midiMenu = ftxui::Menu(&midiPortMenuEntries, &midiPortMenuSelectedIndexLocal, midiMenuOption);
  midiMenu = midiMenu
    | ftxui::Renderer([] (ftxui::Element el) {
        std::unique_lock<std::mutex> lock(m);
        return ftxui::vbox({
            ftxui::text("Please select which midi output you would like to use"),
            ftxui::separator(),
            el,
            ftxui::separator(),
            ftxui::text("Currently selected:") | ftxui::bold,
            ftxui::separator(),
            ftxui::text(midiPortMenuSelectedIndex == std::numeric_limits<int>::max() ? "[none]" : midiPortMenuEntries[midiPortMenuSelectedIndex]),
          }) | ftxui::border;
      })
    | ftxui::Maybe([&] { return mainMenuSelectedIndex == 0; });

  auto noseCalc = ftxui::Button("Ready", [&] {
    m.lock();
    noseZeroCalibrated[0] = noseCurrentPosition[0];
    noseZeroCalibrated[1] = noseCurrentPosition[1];
    noseZeroCalibrationComplete = true;
    m.unlock();
    mainMenuSelectedIndex = 2;
  });
  noseCalc = noseCalc
    | ftxui::Renderer([] (ftxui::Element el) {
        std::unique_lock<std::mutex> lock(m);
        return ftxui::vbox({
            ftxui::text("Please put your nose in the center of the screen in order to calibrate the center of where your face will be."),
            ftxui::text("Please press 'Ready' when ready to save location"),
            ftxui::separator(),
            el,
            ftxui::separator(),
            ftxui::text("Saved location: (" + std::to_string(noseZeroCalibrated[0]) + ", " + std::to_string(noseZeroCalibrated[1]) + ")"),
          }) | ftxui::border;
      })
    | ftxui::Maybe([&] { return mainMenuSelectedIndex == 1; });

  auto mouthCalc1 = ftxui::Button("Ready", [&] {
    calcAverage(&mouthOuterLipUpDownCurrentDistance, &mouthOuterLipUpDownClosedDistanceCalibrated, 100);
    calcAverage(&mouthOuterLipRightLeftCurrentDistance, &mouthOuterLipRightLeftClosedDistanceCalibrated, 100);
    calcAverage(&mouthInnerLipUpDownCurrentDistance, &mouthInnerLipUpDownClosedDistanceCalibrated, 100);
    calcAverage(&mouthInnerLipRightLeftCurrentDistance, &mouthInnerLipRightLeftClosedDistanceCalibrated, 100);
    m.lock();
    mouthCalibrationClosedComplete = true;
    m.unlock();
  });
  mouthCalc1 = mouthCalc1
    | ftxui::Renderer([] (ftxui::Element el) {
        std::unique_lock<std::mutex> lock(m);
        return ftxui::vbox({
            ftxui::text("Please close your mouth in a relaxed manner."),
            ftxui::text("Please press 'Ready' when ready to save location"),
            ftxui::separator(),
            el,
            ftxui::separator(),
            ftxui::text("Saved dimensions: outer up-down:"
                + std::to_string(mouthOuterLipUpDownClosedDistanceCalibrated)
                + "px; inner up-down: "
                + std::to_string(mouthInnerLipUpDownClosedDistanceCalibrated)
                + "px; inner right-left: "
                + std::to_string(mouthInnerLipRightLeftClosedDistanceCalibrated)
                + "px; outer right-left: "
                + std::to_string(mouthOuterLipRightLeftClosedDistanceCalibrated)
                + "px"),
          }) | ftxui::border;
      })
    | ftxui::Maybe([&] { return mainMenuSelectedIndex == 2; });

  auto mouthCalc2 = ftxui::Button("Ready", [&] {
    calcAverage(&mouthOuterLipUpDownCurrentDistance, &mouthOuterLipUpDownOpenDistanceCalibrated, 100);
    calcAverage(&mouthInnerLipUpDownCurrentDistance, &mouthInnerLipUpDownOpenDistanceCalibrated, 100);
    m.lock();
    mouthCalibrationOpenUpDownComplete = true;
    m.unlock();
  });
  mouthCalc2 = mouthCalc2
    | ftxui::Renderer([] (ftxui::Element el) {
        std::unique_lock<std::mutex> lock(m);
        return ftxui::vbox({
            ftxui::text("Please open your mouth as if you were yawning."),
            ftxui::text("Please press 'Ready' when ready to save location"),
            ftxui::separator(),
            el,
            ftxui::separator(),
            ftxui::text("Saved dimensions: outer:" + std::to_string(mouthOuterLipUpDownOpenDistanceCalibrated) + "px; inner: " + std::to_string(mouthInnerLipUpDownOpenDistanceCalibrated) + "px"),
          }) | ftxui::border;
      })
    | ftxui::Maybe([&] { return mainMenuSelectedIndex == 3; });

  auto mouthCalc3 = ftxui::Button("Ready", [&] {
    calcAverage(&mouthOuterLipRightLeftCurrentDistance, &mouthOuterLipRightLeftOpenDistanceCalibrated, 100);
    calcAverage(&mouthInnerLipRightLeftCurrentDistance, &mouthInnerLipRightLeftOpenDistanceCalibrated, 100);
    m.lock();
    mouthCalibrationOpenRightLeftComplete = true;
    m.unlock();
  });
  mouthCalc3 = mouthCalc3
    | ftxui::Renderer([] (ftxui::Element el) {
        std::unique_lock<std::mutex> lock(m);
        return ftxui::vbox({
            ftxui::text("Please smile widely without opening your mouth."),
            ftxui::text("Please press 'Ready' when ready to save location"),
            ftxui::separator(),
            el,
            ftxui::separator(),
            ftxui::text("Saved dimensions: outer:" + std::to_string(mouthOuterLipRightLeftOpenDistanceCalibrated) + "px; inner: " + std::to_string(mouthInnerLipRightLeftOpenDistanceCalibrated) + "px"),
          }) | ftxui::border;
      })
    | ftxui::Maybe([&] { return mainMenuSelectedIndex == 4; });

  auto pitchBoardCalc1 = ftxui::Button("Ready", [&] {
    m.lock();
    nosePitchBoardUpperLeftPosition[0] = (noseCurrentPosition[0] - noseZeroCalibrated[0]) * noseSensitivity + noseZeroCalibrated[0];
    nosePitchBoardUpperLeftPosition[1] = (noseCurrentPosition[1] - noseZeroCalibrated[1]) * noseSensitivity + noseZeroCalibrated[1];
    nosePitchBoardUpperLeftComplete = true;
    if (nosePitchBoardUpperLeftComplete && nosePitchBoardUpperRightComplete && nosePitchBoardLowerRightComplete && nosePitchBoardLowerLeftComplete && pitchBoardNumberOfFrets > 0) {
      m.unlock();
      calculatePitchBoardPoints();
    }
    m.unlock();
  });
  pitchBoardCalc1 = pitchBoardCalc1
    | ftxui::Renderer([] (ftxui::Element el) {
        std::unique_lock<std::mutex> lock(m);
        return ftxui::vbox({
            ftxui::text("Choose a location for the UPPER LEFT corner of the pitch board using your nose."),
            ftxui::text("Please press 'Ready' when ready to save location"),
            ftxui::separator(),
            el,
            ftxui::separator(),
            ftxui::text("Saved location: (" + std::to_string(nosePitchBoardUpperLeftPosition[0]) + ", " + std::to_string(nosePitchBoardUpperLeftPosition[1]) + ")"),
          }) | ftxui::border;
      })
    | ftxui::Maybe([&] { return mainMenuSelectedIndex == 5; });

  auto pitchBoardCalc2 = ftxui::Button("Ready", [&] {
    m.lock();
    nosePitchBoardUpperRightPosition[0] = (noseCurrentPosition[0] - noseZeroCalibrated[0]) * noseSensitivity + noseZeroCalibrated[0];
    nosePitchBoardUpperRightPosition[1] = (noseCurrentPosition[1] - noseZeroCalibrated[1]) * noseSensitivity + noseZeroCalibrated[1];
    nosePitchBoardUpperRightComplete = true;
    if (nosePitchBoardUpperLeftComplete && nosePitchBoardUpperRightComplete && nosePitchBoardLowerRightComplete && nosePitchBoardLowerLeftComplete && pitchBoardNumberOfFrets > 0) {
      m.unlock();
      calculatePitchBoardPoints();
    }
    m.unlock();
  });
  pitchBoardCalc2 = pitchBoardCalc2
    | ftxui::Renderer([] (ftxui::Element el) {
        std::unique_lock<std::mutex> lock(m);
        return ftxui::vbox({
            ftxui::text("Choose a location for the UPPER RIGHT corner of the pitch board using your nose."),
            ftxui::text("Please press 'Ready' when ready to save location"),
            ftxui::separator(),
            el,
            ftxui::separator(),
            ftxui::text("Saved location: (" + std::to_string(nosePitchBoardUpperRightPosition[0]) + ", " + std::to_string(nosePitchBoardUpperRightPosition[1]) + ")"),
          }) | ftxui::border;
      })
    | ftxui::Maybe([&] { return mainMenuSelectedIndex == 6; });
  auto pitchBoardCalc3 = ftxui::Button("Ready", [&] {
    m.lock();
    nosePitchBoardLowerRightPosition[0] = (noseCurrentPosition[0] - noseZeroCalibrated[0]) * noseSensitivity + noseZeroCalibrated[0];
    nosePitchBoardLowerRightPosition[1] = (noseCurrentPosition[1] - noseZeroCalibrated[1]) * noseSensitivity + noseZeroCalibrated[1];
    nosePitchBoardLowerRightComplete = true;
    if (nosePitchBoardUpperLeftComplete && nosePitchBoardUpperRightComplete && nosePitchBoardLowerRightComplete && nosePitchBoardLowerLeftComplete && pitchBoardNumberOfFrets > 0) {
      m.unlock();
      calculatePitchBoardPoints();
    }
    m.unlock();
  });
  pitchBoardCalc3 = pitchBoardCalc3
    | ftxui::Renderer([] (ftxui::Element el) {
        std::unique_lock<std::mutex> lock(m);
        return ftxui::vbox({
            ftxui::text("Choose a location for the LOWER RIGHT corner of the pitch board using your nose."),
            ftxui::text("Please press 'Ready' when ready to save location"),
            ftxui::separator(),
            el,
            ftxui::separator(),
            ftxui::text("Saved location: (" + std::to_string(nosePitchBoardLowerRightPosition[0]) + ", " + std::to_string(nosePitchBoardLowerRightPosition[1]) + ")"),
          }) | ftxui::border;
      })
    | ftxui::Maybe([&] { return mainMenuSelectedIndex == 7; });
  auto pitchBoardCalc4 = ftxui::Button("Ready", [&] {
    m.lock();
    nosePitchBoardLowerLeftPosition[0] = (noseCurrentPosition[0] - noseZeroCalibrated[0]) * noseSensitivity + noseZeroCalibrated[0];
    nosePitchBoardLowerLeftPosition[1] = (noseCurrentPosition[1] - noseZeroCalibrated[1]) * noseSensitivity + noseZeroCalibrated[1];
    nosePitchBoardLowerLeftComplete = true;
    if (nosePitchBoardUpperLeftComplete && nosePitchBoardUpperRightComplete && nosePitchBoardLowerRightComplete && nosePitchBoardLowerLeftComplete && pitchBoardNumberOfFrets > 0) {
      m.unlock();
      calculatePitchBoardPoints();
    }
    m.unlock();
  });
  pitchBoardCalc4 = pitchBoardCalc4
    | ftxui::Renderer([] (ftxui::Element el) {
        std::unique_lock<std::mutex> lock(m);
        return ftxui::vbox({
            ftxui::text("Choose a location for the LOWER LEFT corner of the pitch board using your nose."),
            ftxui::text("Please press 'Ready' when ready to save location"),
            ftxui::separator(),
            el,
            ftxui::separator(),
            ftxui::text("Saved location: (" + std::to_string(nosePitchBoardLowerLeftPosition[0]) + ", " + std::to_string(nosePitchBoardLowerLeftPosition[1]) + ")"),
          }) | ftxui::border;
      })
    | ftxui::Maybe([&] { return mainMenuSelectedIndex == 8; });

  std::string pitchBoardNumberOfFretsStr;
  ftxui::InputOption inputOption;
  inputOption.on_enter = [&pitchBoardNumberOfFretsStr] {
    try {
      m.lock();
      pitchBoardNumberOfFrets = (unsigned int)std::stoi(pitchBoardNumberOfFretsStr);
      if (nosePitchBoardUpperLeftComplete && nosePitchBoardUpperRightComplete && nosePitchBoardLowerRightComplete && nosePitchBoardLowerLeftComplete && pitchBoardNumberOfFrets > 0) {
        m.unlock();
        calculatePitchBoardPoints();
      }
      m.unlock();
    } catch (exception& e) {
    }
  };
  auto pitchBoardCalc5 = ftxui::Input(&pitchBoardNumberOfFretsStr, "Number of frets", inputOption);
  pitchBoardCalc5 = pitchBoardCalc5
    | ftxui::Renderer([] (ftxui::Element el) {
      std::unique_lock<std::mutex> lock(m);
      return ftxui::vbox({
        ftxui::text("Input how many 'frets' to have on your pitch board. (think violin fretboard)"),
        ftxui::separator(),
        el,
        ftxui::separator(),
        ftxui::text("Saved value: " + std::to_string(pitchBoardNumberOfFrets)),
      }) | ftxui::border;
    })
    | ftxui::Maybe([&] { return mainMenuSelectedIndex == 9; });

  auto canvas = ftxui::Renderer([] () {
      auto c = ftxui::Canvas(cvMatrixWidth, cvMatrixHeight);
      if (faceDetected.load()) {
        // obtain mutex lock in this context, so it releases ownership on context exit
        std::unique_lock<std::mutex> lock(m);
        // draw out pitch board when all necessary parameters are configured
        if (true
          && nosePitchBoardUpperLeftComplete
          && nosePitchBoardUpperRightComplete
          && nosePitchBoardLowerRightComplete
          && nosePitchBoardLowerLeftComplete
          && pitchBoardNumberOfFrets > 0) {
          if (nosePitchBoardCalculateComplete) {
            // connect all outer points
            for (int i = 0; i <= pitchBoardNumberOfFrets; i++) {
              c.DrawPointLine(pitchBoardPoints[i][0][0], pitchBoardPoints[i][0][1], pitchBoardPoints[i][pitchBoardNumberOfStrings][0], pitchBoardPoints[i][pitchBoardNumberOfStrings][1]);
            }
            for (int j = 0; j <= pitchBoardNumberOfStrings; j++) {
              c.DrawPointLine(pitchBoardPoints[0][j][0], pitchBoardPoints[0][j][1], pitchBoardPoints[pitchBoardNumberOfFrets][j][0], pitchBoardPoints[pitchBoardNumberOfFrets][j][1]);
            }
            // FIXME: DEBUG draw all points
            for (int i = 0; i <= pitchBoardNumberOfFrets; i++) {
              for (int j = 0; j <= pitchBoardNumberOfStrings; j++) {
                c.DrawPointCircleFilled(pitchBoardPoints[i][j][0], pitchBoardPoints[i][j][1], 1, ftxui::Color::Red);
              }
            }
          }
        } else {
          for (int i = 0; i < 68; i++) {
            c.DrawPointCircleFilled(dlibFacialParts[i][0], dlibFacialParts[i][1], 1, ftxui::Color::Green);
          }
          // draw specified corners of the pitch board because it's not all mapped out yet
          if (nosePitchBoardUpperLeftComplete) {
            c.DrawPointCircleFilled(nosePitchBoardUpperLeftPosition[0], nosePitchBoardUpperLeftPosition[1], 1, ftxui::Color::Red);
          }
          if (nosePitchBoardUpperRightComplete) {
            c.DrawPointCircleFilled(nosePitchBoardUpperRightPosition[0], nosePitchBoardUpperRightPosition[1], 1, ftxui::Color::Red);
          }
          if (nosePitchBoardLowerRightComplete) {
            c.DrawPointCircleFilled(nosePitchBoardLowerRightPosition[0], nosePitchBoardLowerRightPosition[1], 1, ftxui::Color::Red);
          }
          if (nosePitchBoardLowerLeftComplete) {
            c.DrawPointCircleFilled(nosePitchBoardLowerLeftPosition[0], nosePitchBoardLowerLeftPosition[1], 1, ftxui::Color::Red);
          }
        }
        // draw guide lines on the edges of the canvas to show how open the mouth is
        if (true
          && mouthCalibrationClosedComplete
          && mouthCalibrationOpenUpDownComplete
          && mouthCalibrationOpenRightLeftComplete) {
          c.DrawPointLine(
              cvMatrixWidth - 1, round(cvMatrixHeight / 2) - round(mouthOpenPercentage * (float)cvMatrixHeight / 4.0),
              cvMatrixWidth - 1 - 5, round(cvMatrixHeight / 2) - round(mouthOpenPercentage * (float)cvMatrixHeight / 4.0));
          c.DrawPointLine(
              cvMatrixWidth - 1, round(cvMatrixHeight / 2) + round(mouthOpenPercentage * (float)cvMatrixHeight / 4.0),
              cvMatrixWidth - 1 - 5, round(cvMatrixHeight / 2) + round(mouthOpenPercentage * (float)cvMatrixHeight / 4.0));
          c.DrawPointLine(
              round(cvMatrixWidth / 2) - round(mouthWidePercentage * (float)cvMatrixWidth / 4.0), cvMatrixHeight - 1,
              round(cvMatrixWidth / 2) - round(mouthWidePercentage * (float)cvMatrixWidth / 4.0), cvMatrixHeight - 1 - 5);
          c.DrawPointLine(
              round(cvMatrixWidth / 2) + round(mouthWidePercentage * (float)cvMatrixWidth / 4.0), cvMatrixHeight - 1,
              round(cvMatrixWidth / 2) + round(mouthWidePercentage * (float)cvMatrixWidth / 4.0), cvMatrixHeight - 1 - 5);
        } else {
        }
        if (noseZeroCalibrationComplete) {
          c.DrawPointCircleFilled(noseCurrentPositionMultiplied[0], noseCurrentPositionMultiplied[1], 1);
        } else {
          c.DrawPointCircleFilled(noseCurrentPosition[0], noseCurrentPosition[1], 1);
        }
        return ftxui::vbox({
          ftxui::hbox({
            ftxui::canvas(std::move(c)) | ftxui::border,
            ftxui::gaugeUp(mouthOpenPercentage),
          }),
          ftxui::gaugeRight(mouthWidePercentage),
        });
      } else {
        return ftxui::text("No face detected.") | ftxui::border;
      }
    });

  auto componentLayout = ftxui::Container::Vertical({
    midiMenu,
    noseCalc,
    mouthCalc1,
    mouthCalc2,
    mouthCalc3,
    pitchBoardCalc1,
    pitchBoardCalc2,
    pitchBoardCalc3,
    pitchBoardCalc4,
    pitchBoardCalc5,
  });
  auto layout = ftxui::Container::Horizontal({
    mainMenu,
    componentLayout,
    canvas,
  });
  auto renderer = ftxui::Renderer(layout, [&] {
      return ftxui::hbox({
        mainMenu->Render(),
        ftxui::separator(),
        ftxui::filler(),
        ftxui::vbox({
          ftxui::hbox({
            ftxui::filler(),
            componentLayout->Render(),
            ftxui::filler(),
          }),
          ftxui::filler(),
          ftxui::hbox({
            ftxui::filler(),
            canvas->Render(),
            ftxui::filler(),
          }),
          ftxui::filler(),
        }),
        ftxui::filler(),
      });
    });
  renderer = renderer
    | ftxui::CatchEvent([&] (ftxui::Event event) {
        if (event == ftxui::Event::Character('q')) {
          run.store(false);
          screen.ExitLoopClosure()();
          return true;
        }
        return false;
      });
  // spinlock until greenlight is given
  while (run.load()) {
    m.lock();
    if (calibrationGreenLight) {
      m.unlock();
      break;
    }
    m.unlock();
    std::this_thread::sleep_for(100ms);
  }
  screen.Loop(renderer);
}

void midiDriver() {
  // initialize polyphonyCache variable
  polyphonyCache = new polyphonyCacheItem[polyphonyCacheSize];
  // https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
  libremidi::midi_out midi = libremidi::midi_out(libremidi::API::LINUX_ALSA_SEQ, "test");
  // enumerate midi ports to be displayed to user
  midiMutex.lock();
  for (int i = 0, N = midi.get_port_count(); i < N; i++) {
    midiPortMenuEntries.push_back(midi.get_port_name(i));
  }
  midiMutex.unlock();
  // tell other threads midi ports are populated
  midiPortsPopulated.store(true);
  // wait for midi port to be chosen
  while (!midiPortSelected.load()) {
    if (!run.load()) {
      return;
    }
  }
  midi.open_port(midiPortMenuSelectedIndex);
  // startup sequence
  std::this_thread::sleep_for(1000ms);
  midi.send_message(libremidi::message::note_on((uint8_t)1, (uint8_t)48, (uint8_t)70));
  std::this_thread::sleep_for(100ms);
  midi.send_message(libremidi::message::note_on((uint8_t)1, (uint8_t)52, (uint8_t)70));
  std::this_thread::sleep_for(100ms);
  midi.send_message(libremidi::message::note_on((uint8_t)1, (uint8_t)55, (uint8_t)70));
  std::this_thread::sleep_for(100ms);
  midi.send_message(libremidi::message::note_on((uint8_t)1, (uint8_t)59, (uint8_t)70));
  std::this_thread::sleep_for(1000ms);
  midi.send_message(libremidi::message::note_on((uint8_t)1, (uint8_t)78, (uint8_t)70));
  std::this_thread::sleep_for(1200ms);
  midi.send_message(libremidi::message::note_on((uint8_t)1, (uint8_t)74, (uint8_t)55));
  std::this_thread::sleep_for(1200ms);
  midi.send_message(libremidi::message::note_off((uint8_t)1, (uint8_t)48, (uint8_t)70));
  midi.send_message(libremidi::message::note_off((uint8_t)1, (uint8_t)52, (uint8_t)70));
  midi.send_message(libremidi::message::note_off((uint8_t)1, (uint8_t)55, (uint8_t)70));
  midi.send_message(libremidi::message::note_off((uint8_t)1, (uint8_t)59, (uint8_t)70));
  midi.send_message(libremidi::message::note_off((uint8_t)1, (uint8_t)78, (uint8_t)70));
  midi.send_message(libremidi::message::note_off((uint8_t)1, (uint8_t)74, (uint8_t)55));
  uint8_t channel;
  uint8_t note;
  uint8_t velocity;
  std::chrono::high_resolution_clock::time_point timePoint;
  unsigned int polyphonyCacheIndex = 0;
  while (run.load()) {
    m.lock();
    if (true
      && noseZeroCalibrationComplete
      && nosePitchBoardCalculateComplete
      && mouthCalibrationClosedComplete
      && mouthCalibrationOpenUpDownComplete
      && mouthCalibrationOpenRightLeftComplete) {
      m.unlock();
      break;
    }
    m.unlock();
    std::this_thread::sleep_for(100ms);
  }
  while (run.load()) {
    midiMutex.lock();
    bool debounceIgnoreNote = false;
    if (false) {
    } else if (mouthOpen && !midiPlaying) {
      channel = 1;
      note = midiNoteToPlay;
      velocity = round(127 * (1.0 - mouthWidePercentage));
      timePoint = std::chrono::high_resolution_clock::now();
      // debounce input for this note
      for (int i = 0; i < polyphonyCacheSize; i++) {
        if (polyphonyCache[i].note == note) {
          if (std::chrono::duration_cast<std::chrono::milliseconds>(timePoint - polyphonyCache[i].entryTime).count() > 80) {
            // if last time played is above debounce threshold, turn note off so it can be played again
            if (polyphonyCache[polyphonyCacheIndex].note == note) {
              midi.send_message(libremidi::message::note_off(channel, note, velocity));
            }
          } else {
            // if last time played is less than/equal to debounce threshold, ignore this event
            debounceIgnoreNote = true;
          }
          break;
        }
      }
      if (!debounceIgnoreNote) {
        polyphonyCache[polyphonyCacheIndex].note = note;
        polyphonyCache[polyphonyCacheIndex].entryTime = timePoint;
        midi.send_message(libremidi::message::note_on(channel, note, velocity));
        midiPlaying = true;
        polyphonyCacheIndex = (polyphonyCacheIndex + 1) % polyphonyCacheSize;
        debounceIgnoreNote = false;
      }
    } else if (!mouthOpen && midiPlaying) {
      channel = 1;
      note = midiNoteToPlay;
      velocity = 127;
      for (int i = 0; i < polyphonyCacheSize; i++) {
        if (polyphonyCache[i].note == note) {
          // remove note; we stop playing it
          polyphonyCache[i].note = std::numeric_limits<int>::max();
          polyphonyCache[i].entryTime = std::chrono::high_resolution_clock::time_point::min();
        }
      }
      midi.send_message(libremidi::message::note_off(channel, note, velocity));
      midiPlaying = false;
    }
    for (int i = 0; i < polyphonyCacheSize; i++) {
      timePoint = std::chrono::high_resolution_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(timePoint - polyphonyCache[i].entryTime).count() > 4000) {
        polyphonyCache[i].note == std::numeric_limits<int>::max();
        polyphonyCache[i].entryTime == std::chrono::high_resolution_clock::time_point::min();
      }
    }
    midiMutex.unlock();
  }
}

int main(int argc, char** argv) {
  try {
    std::thread thread_tuiRenderer(tuiRenderer);
    std::thread thread_interpretFacialData(interpretFacialData);
    std::thread thread_midiDriver(midiDriver);

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, capPropFrameWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, capPropFrameHeight);
    if (!cap.isOpened()) {
      cerr << "Unable to connect to camera" << endl;
      return 1;
    }
    if (argc == 1) {
      cout << "Call this program like this:" << endl;
      cout << "./program /path/to/shape_predictor_68_face_landmarks.dat" << endl;
      cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
      cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
      return 0;
    }

    // get bounding boxes for each face in the image
    frontal_face_detector detector = get_frontal_face_detector();
    // the shape_predictor predicts face landmark positions given an image and face bounding box
    // will need to download shape_predictor_68_face_landmarks.dat.bz2 before using
    shape_predictor sp;
    deserialize(argv[1]) >> sp;


    // FIXME iamge_window: image_window win;
    std::vector<rectangle> faces;
    while (run.load()) {
      m.lock();
      calibrationGreenLight = true;
      m.unlock();
      cv::Mat cvMatrix;
      if (!cap.read(cvMatrix)) {
        break;
      }
      // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
      // wraps the Mat object, it doesn't copy anything.  So baseimg is only valid as
      // long as cvMatrix is valid.  Also don't do anything to cvMatrix that would cause it
      // to reallocate the memory which stores the image as that will make baseimg
      // contain dangling pointers.  This basically means you shouldn't modify matrix
      // while using baseimg.

      // set color to grayscale
      cv::cvtColor(cvMatrix, cvMatrix, cv::COLOR_BGR2GRAY);
      cv::flip(cvMatrix, cvMatrix, 1);
      // use unsigned char instead of `bgr_pixel` as pixel type,
      // because we are working with grayscale magnitude
      cv_image<unsigned char> baseimg(cvMatrix);
      // detect faces
      faces = detector(baseimg);
      // find the pose of each face
      // assume only one face is detected, because only one person will be using this
      if (!faces.empty()) {
        faceDetected.store(true);
        full_object_detection shape = sp(baseimg, faces[0]);

        m.lock();
        // draw out pitch board when all necessary parameters are configured
        if (true
          && nosePitchBoardUpperLeftComplete
          && nosePitchBoardUpperRightComplete
          && nosePitchBoardLowerRightComplete
          && nosePitchBoardLowerLeftComplete
          && pitchBoardNumberOfFrets > 0) {
        } else {
          // store facial parts globally to be drawn during configuration
          for (int i = 0; i < 68; i++) {
            dlibFacialParts[i][0] = shape.part(i).x();
            dlibFacialParts[i][1] = shape.part(i).y();
          }
        }

        // save the cvMatrix's size in order to know the size of our 'canvas' in calculation
        cvMatrixWidth = cvMatrix.cols;
        cvMatrixHeight = cvMatrix.rows;

        // capture nose position
        noseCurrentPosition[0] = shape.part(31-1).x();
        noseCurrentPosition[1] = shape.part(31-1).y();
        if (noseZeroCalibrationComplete) {
          noseCurrentPositionMultiplied[0] = (noseCurrentPosition[0] - noseZeroCalibrated[0]) * noseSensitivity + noseZeroCalibrated[0];
          noseCurrentPositionMultiplied[1] = (noseCurrentPosition[1] - noseZeroCalibrated[1]) * noseSensitivity + noseZeroCalibrated[1];
        }
        // capture mouth dimensions
        mouthOuterLipUpDownCurrentDistance = shape.part(58-1).y() - shape.part(52-1).y();
        mouthOuterLipRightLeftCurrentDistance = shape.part(55-1).x() - shape.part(49-1).x();
        m.unlock();

        // trigger render in TUI
        screen.PostEvent(ftxui::Event::Custom);
      } else {
        faceDetected.store(false);
        // trigger render in TUI
        screen.PostEvent(ftxui::Event::Custom);
      }
    }
    thread_tuiRenderer.join();
    thread_interpretFacialData.join();
    thread_midiDriver.join();
    return 0;
  }
  catch(serialization_error& e) {
    cout << "You need dlib's default face landmarking model file to run this example." << endl;
    cout << "You can get it from the following URL: " << endl;
    cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
    cout << endl << e.what() << endl;
  }
  catch (exception& e) {
    cout << "\nexception thrown!" << endl;
    cout << e.what() << endl;
  }
}
