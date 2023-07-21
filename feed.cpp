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
#include <mutex>
#include <math.h>
#include <chrono>
#include <ctime>
#include <libremidi/libremidi.hpp>

using namespace dlib;
using namespace std;

#if DEBUG
static const bool isDebug = true;
#else
static const bool isDebug = false;
#endif

std::mutex m;
std::mutex midiMutex;

const unsigned int capPropFrameWidth = 100;
const unsigned int capPropFrameHeight = 100;

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
bool TEMPpitchBoardLatticeCalculated = false;
float*** pitchBoardPointsAsFloats;
unsigned int*** pitchBoardPoints;
struct pitchBoardQuadrilateralCacheItem {
  float area;
  unsigned char midiNote;
};
pitchBoardQuadrilateralCacheItem** pitchBoardQuadrilateralCache;

bool midiPlaying;
bool mouthOpen;
float mouthOpenPercentage;
float mouthWidePercentage;
unsigned int scaleDegree;
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

// stab at seeing if eye tracking was a viable option for getting information. Turns out, it was not. At least, for now.
void eyeTracking(dlib::image_window* win, cv::Mat* mat, dlib::full_object_detection* shape) {
  cv::Size matrixSize = mat->size();
  int matrixType = mat->type();
  // mask used to block out area where eyes are located
  cv::Mat eyeMaskMatrix = cv::Mat::zeros(matrixSize, matrixType);
  std::vector<cv::Point> leftEyePoints = {
    cv::Point(shape->part(37 - 1).x(), shape->part(37 - 1).y()),
    cv::Point(shape->part(38 - 1).x(), shape->part(38 - 1).y()),
    cv::Point(shape->part(39 - 1).x(), shape->part(39 - 1).y()),
    cv::Point(shape->part(40 - 1).x(), shape->part(40 - 1).y()),
    cv::Point(shape->part(41 - 1).x(), shape->part(41 - 1).y()),
    cv::Point(shape->part(42 - 1).x(), shape->part(42 - 1).y()),
  };
  std::vector<cv::Point> rightEyePoints = {
    cv::Point(shape->part(43 - 1).x(), shape->part(43 - 1).y()),
    cv::Point(shape->part(44 - 1).x(), shape->part(44 - 1).y()),
    cv::Point(shape->part(45 - 1).x(), shape->part(45 - 1).y()),
    cv::Point(shape->part(46 - 1).x(), shape->part(46 - 1).y()),
    cv::Point(shape->part(47 - 1).x(), shape->part(47 - 1).y()),
    cv::Point(shape->part(48 - 1).x(), shape->part(48 - 1).y())
  };
  // use points from dlib's detection to fill in black polygons where eyes are
  cv::fillConvexPoly(eyeMaskMatrix, leftEyePoints, cv::Scalar(255, 255, 255));
  cv::fillConvexPoly(eyeMaskMatrix, rightEyePoints, cv::Scalar(255, 255, 255));
  // 'kernel' used for erosion/dilation algorithm
  cv::Mat kernel = cv::Mat::ones(5, 5, matrixType);
  // use image dilation to expand the borders of the eye regions
  cv::dilate(eyeMaskMatrix, eyeMaskMatrix, kernel);

  // used to overlay at the end to allow opencv to find contours better within the eye shapes
  cv::Mat eyeBoundaryMatrix = eyeMaskMatrix.clone();
  cv::Mat eyeBoundaryInnerMatrix = eyeMaskMatrix.clone();
  cv::erode(eyeBoundaryInnerMatrix, eyeBoundaryInnerMatrix, cv::Mat());
  cv::bitwise_xor(eyeBoundaryMatrix, eyeBoundaryInnerMatrix, eyeBoundaryMatrix, eyeMaskMatrix);
  cv::bitwise_not(eyeBoundaryMatrix, eyeBoundaryMatrix);

  // matrix of actual eyes in image
  cv::Mat eyesMatrix = cv::Mat::zeros(matrixSize, matrixType);
  // get a cropping of just the eyes using the masking and output it to eyeMatrix
  cv::bitwise_and(*mat, *mat, eyesMatrix, eyeMaskMatrix);
  // find mean of values in BW eyesMatrix
  cv::Scalar eyesMatrixMean = cv::mean(eyesMatrix, eyeMaskMatrix);
  // use threshold value to be a function of eyesMatrixMean in order to make it responsive to lighting changes
  unsigned char threshold = floor(eyesMatrixMean[0]) - 10;
  cv::threshold(eyesMatrix, eyesMatrix, threshold, 255, cv::THRESH_BINARY_INV);
  cv::Mat pupilKernel = cv::Mat::ones(3, 3, matrixType);
  cv::erode(eyesMatrix, eyesMatrix, pupilKernel); // shave off imperfections
  // cv::dilate(eyesMatrix, eyesMatrix, cv::Mat()); // blow up smoother shapes

  cv::bitwise_and(eyeBoundaryMatrix, eyesMatrix, eyesMatrix);
  // { deletable
  opencvShowGrayscaleMatrix(win, &eyesMatrix);
  // } deletable
}

void interpretFacialData() {
  // wait for calibration before running
  while (true) {
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
  while (true) {
    // FIXME: determine which note to play
    bool boundingBoxFound = false;
    for (int i = 0; i < pitchBoardNumberOfFrets; i++) {
      if (boundingBoxFound)
        break;
      for (int j = 0; j < pitchBoardNumberOfStrings; j++) {
        if (boundingBoxFound)
          break;
        // quadrilateral area using center point
        float quadrilateralAreaToCompare =
          triangleArea(
              noseCurrentPositionMultiplied[0],
              noseCurrentPositionMultiplied[1],
              (int)pitchBoardPoints[i][j][0],
              (int)pitchBoardPoints[i][j][1],
              (int)pitchBoardPoints[i+1][j][0],
              (int)pitchBoardPoints[i+1][j][1])
          + triangleArea(noseCurrentPositionMultiplied[0],
              noseCurrentPositionMultiplied[1],
              (int)pitchBoardPoints[i+1][j][0],
              (int)pitchBoardPoints[i+1][j][1],
              (int)pitchBoardPoints[i+1][j+1][0],
              (int)pitchBoardPoints[i+1][j+1][1])
          + triangleArea(noseCurrentPositionMultiplied[0],
              noseCurrentPositionMultiplied[1],
              (int)pitchBoardPoints[i+1][j+1][0],
              (int)pitchBoardPoints[i+1][j+1][1],
              (int)pitchBoardPoints[i][j+1][0],
              (int)pitchBoardPoints[i][j+1][1])
          + triangleArea(noseCurrentPositionMultiplied[0],
              noseCurrentPositionMultiplied[1],
              (int)pitchBoardPoints[i][j+1][0],
              (int)pitchBoardPoints[i][j+1][1],
              (int)pitchBoardPoints[i][j][0],
              (int)pitchBoardPoints[i][j][1]);
        if (quadrilateralAreaToCompare == pitchBoardQuadrilateralCache[i][j].area) {
          cout << "(" << i << ", " << j << ")" << endl;
          boundingBoxFound = true;
        }
      }
    }
    // handle mouth dimensions
    m.lock();
    if (mouthOuterLipUpDownCurrentDistance > mouthOuterLipUpDownClosedDistanceCalibrated) {
      midiMutex.lock();
      mouthOpen = true;
      mouthOpenPercentage = percentageNormalize((float)(
          (float)((mouthOuterLipUpDownCurrentDistance - mouthOuterLipUpDownClosedDistanceCalibrated + mouthInnerLipUpDownCurrentDistance - mouthInnerLipUpDownClosedDistanceCalibrated) / 2.0)
          /
          (float)((mouthOuterLipUpDownOpenDistanceCalibrated - mouthOuterLipUpDownClosedDistanceCalibrated + mouthInnerLipUpDownOpenDistanceCalibrated - mouthInnerLipUpDownClosedDistanceCalibrated) / 2.0)
      ));
      mouthWidePercentage = percentageNormalize((float)(
          (float)((mouthOuterLipRightLeftCurrentDistance - mouthOuterLipRightLeftClosedDistanceCalibrated + mouthInnerLipRightLeftCurrentDistance - mouthInnerLipRightLeftClosedDistanceCalibrated) / 2.0)
          /
          (float)((mouthOuterLipRightLeftOpenDistanceCalibrated - mouthOuterLipRightLeftClosedDistanceCalibrated + mouthInnerLipRightLeftOpenDistanceCalibrated - mouthInnerLipRightLeftClosedDistanceCalibrated) / 2.0)
      ));
      midiMutex.unlock();
    } else {
      midiMutex.lock();
      mouthOpen = false;
      midiMutex.unlock();
    }
    m.unlock();
  }
}

void calculatePitchBoardPoints() {
  // calculate reduced row eschelon form.
  // Stolen from: https://stackoverflow.com/questions/31756413/solving-a-simple-matrix-in-row-reduced-form-in-c
  // FIXME
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
    cout << "------------------------------" << endl;;
    cout << matrix[0][0] << " " << matrix[0][1] << " " << matrix[0][2] << endl;
    cout << matrix[1][0] << " " << matrix[1][1] << " " << matrix[1][2] << endl;
    cout << "------------------------------" << endl;;
    return parametricLine(p1, p2, matrix[0][2]);
    if (true
      && round(parametricLine(p1, p2, matrix[0][2])[0]) == round(parametricLine(q1, q2, matrix[1][2])[0])
      && round(parametricLine(p1, p2, matrix[0][2])[1]) == round(parametricLine(q1, q2, matrix[1][2])[1])) {
      return parametricLine(p1, p2, matrix[0][2]);
    } else {
      cerr << "error: could not find coincident point of two lines." << endl;
      exit(-1);
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
  // allocate a cache that relates to each 'note' on the pitch board
  pitchBoardQuadrilateralCache = (pitchBoardQuadrilateralCacheItem**)malloc(sizeof(pitchBoardQuadrilateralCacheItem*) * pitchBoardNumberOfFrets);
  // calculate area of all of the quadrilaterals, with ranges x: [0, pitchBoardNumberOfFrets - 1], and y: [0, pitchBoardNumberOfStrings - 1]
  for (int i = 0; i < pitchBoardNumberOfFrets; i++) {
    pitchBoardQuadrilateralCache[i] = (pitchBoardQuadrilateralCacheItem*)malloc(sizeof(pitchBoardQuadrilateralCacheItem) * pitchBoardNumberOfStrings);
    for (int j = 0; j < pitchBoardNumberOfStrings; j++) {
      // calculate area by breaking quadrilateral into two triangles
      // first triangle    second triangle
      // *   *             *
      //     *             *   *
      pitchBoardQuadrilateralCache[i][j].area =
        triangleArea(pitchBoardPoints[i][j], pitchBoardPoints[i + 1][j],pitchBoardPoints[i + 1][j + 1])
        + triangleArea(pitchBoardPoints[i][j], pitchBoardPoints[i][j + 1], pitchBoardPoints[i + 1][j + 1]);
    }
  }
  nosePitchBoardCalculateComplete = true;
  m.unlock();
}

void calibration() {
  int N = 100;
  int measurements[N];
  int acc;
  // calc average of live measured value
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
  // wait for facial recog to set up before running
  while (true) {
    m.lock();
    if (calibrationGreenLight) {
      m.unlock();
      break;
    }
    m.unlock();
    std::this_thread::sleep_for(100ms);
  }
  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#        Nose Calibration        #" << endl;
  cout << "#        ----------------        #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Please put your nose in the   #" << endl;
  cout << "#  center of the screen in       #" << endl;
  cout << "#  order to calibrate the center #" << endl;
  cout << "#  of your pitch wheel.          #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Press [Enter] when ready.     #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  getchar();

  m.lock();
  noseZeroCalibrated[0] = noseCurrentPosition[0];
  noseZeroCalibrated[1] = noseCurrentPosition[1];
  noseZeroCalibrationComplete = true;
  m.unlock();

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#        Position saved!         #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#       Mouth Calibration        #" << endl;
  cout << "#          Step 1 of 3           #" << endl;
  cout << "#       -----------------        #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Please close your mouth       #" << endl;
  cout << "#  in relaxed manner.            #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Press [Enter] when ready.     #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  getchar();

  calcAverage(&mouthOuterLipUpDownCurrentDistance, &mouthOuterLipUpDownClosedDistanceCalibrated, 100);
  calcAverage(&mouthOuterLipRightLeftCurrentDistance, &mouthOuterLipRightLeftClosedDistanceCalibrated, 100);
  calcAverage(&mouthInnerLipUpDownCurrentDistance, &mouthInnerLipUpDownClosedDistanceCalibrated, 100);
  calcAverage(&mouthInnerLipRightLeftCurrentDistance, &mouthInnerLipRightLeftClosedDistanceCalibrated, 100);
  m.lock();
  mouthCalibrationClosedComplete = true;
  m.unlock();

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#        Position saved!         #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#       Mouth Calibration        #" << endl;
  cout << "#          Step 2 of 3           #" << endl;
  cout << "#       -----------------        #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Please open your mouth        #" << endl;
  cout << "#  as if you were yawning.       #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Press [Enter] when ready.     #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  getchar();

  calcAverage(&mouthOuterLipUpDownCurrentDistance, &mouthOuterLipUpDownOpenDistanceCalibrated, 100);
  calcAverage(&mouthInnerLipUpDownCurrentDistance, &mouthInnerLipUpDownOpenDistanceCalibrated, 100);
  m.lock();
  mouthCalibrationOpenUpDownComplete = true;
  m.unlock();

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#        Position saved!         #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#       Mouth Calibration        #" << endl;
  cout << "#          Step 3 of 3           #" << endl;
  cout << "#       -----------------        #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Please smile widely           #" << endl;
  cout << "#  without opening your mouth.   #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Press [Enter] when ready.     #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  getchar();

  calcAverage(&mouthOuterLipRightLeftCurrentDistance, &mouthOuterLipRightLeftOpenDistanceCalibrated, 100);
  calcAverage(&mouthInnerLipRightLeftCurrentDistance, &mouthInnerLipRightLeftOpenDistanceCalibrated, 100);
  m.lock();
  mouthCalibrationOpenRightLeftComplete = true;
  m.unlock();

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#        Position saved!         #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#    Pitch Board Calibration     #" << endl;
  cout << "#          Step 1 of 5           #" << endl;
  cout << "#       -----------------        #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Choose a location for the     #" << endl;
  cout << "#  UPPER LEFT corner of the      #" << endl;
  cout << "#  pitch board.                  #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Press [Enter] when ready.     #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  getchar();

  m.lock();
  nosePitchBoardUpperLeftPosition[0] = (noseCurrentPosition[0] - noseZeroCalibrated[0]) * noseSensitivity + noseZeroCalibrated[0];
  nosePitchBoardUpperLeftPosition[1] = (noseCurrentPosition[1] - noseZeroCalibrated[1]) * noseSensitivity + noseZeroCalibrated[1];
  nosePitchBoardUpperLeftComplete = true;
  m.unlock();

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#        Position saved!         #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#    Pitch Board Calibration     #" << endl;
  cout << "#          Step 2 of 5           #" << endl;
  cout << "#       -----------------        #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Choose a location for the     #" << endl;
  cout << "#  UPPER RIGHT corner of the     #" << endl;
  cout << "#  pitch board.                  #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Press [Enter] when ready.     #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  getchar();

  m.lock();
  nosePitchBoardUpperRightPosition[0] = (noseCurrentPosition[0] - noseZeroCalibrated[0]) * noseSensitivity + noseZeroCalibrated[0];
  nosePitchBoardUpperRightPosition[1] = (noseCurrentPosition[1] - noseZeroCalibrated[1]) * noseSensitivity + noseZeroCalibrated[1];
  nosePitchBoardUpperRightComplete = true;
  m.unlock();

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#        Position saved!         #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#    Pitch Board Calibration     #" << endl;
  cout << "#          Step 3 of 5           #" << endl;
  cout << "#       -----------------        #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Choose a location for the     #" << endl;
  cout << "#  LOWER RIGHT corner of the     #" << endl;
  cout << "#  pitch board.                  #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Press [Enter] when ready.     #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  getchar();

  m.lock();
  nosePitchBoardLowerRightPosition[0] = (noseCurrentPosition[0] - noseZeroCalibrated[0]) * noseSensitivity + noseZeroCalibrated[0];
  nosePitchBoardLowerRightPosition[1] = (noseCurrentPosition[1] - noseZeroCalibrated[1]) * noseSensitivity + noseZeroCalibrated[1];
  nosePitchBoardLowerRightComplete = true;
  m.unlock();

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#        Position saved!         #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#    Pitch Board Calibration     #" << endl;
  cout << "#          Step 4 of 5           #" << endl;
  cout << "#       -----------------        #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Choose a location for the     #" << endl;
  cout << "#  LOWER LEFT corner of the      #" << endl;
  cout << "#  pitch board.                  #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Press [Enter] when ready.     #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  getchar();

  m.lock();
  nosePitchBoardLowerLeftPosition[0] = (noseCurrentPosition[0] - noseZeroCalibrated[0]) * noseSensitivity + noseZeroCalibrated[0];
  nosePitchBoardLowerLeftPosition[1] = (noseCurrentPosition[1] - noseZeroCalibrated[1]) * noseSensitivity + noseZeroCalibrated[1];
  nosePitchBoardLowerLeftComplete = true;
  m.unlock();

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#        Position saved!         #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#    Pitch Board Calibration     #" << endl;
  cout << "#          Step 5 of 5           #" << endl;
  cout << "#       -----------------        #" << endl;
  cout << "#                                #" << endl;
  cout << "#  Input how many 'frets' to     #" << endl;
  cout << "#  have on your pitch board.     #" << endl;
  cout << "#  (think violin fretboard)      #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;

  unsigned int tmp;
  cout << "Number of frets: ";
  cin >> tmp;
  m.lock();
  pitchBoardNumberOfFrets = tmp;
  m.unlock();

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#        Position saved!         #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;

  calculatePitchBoardPoints();

  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#     Calibration complete!      #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
}

void midiDriver() {
  return; // FIXME: for debug
  // initialize polyphonyCache variable
  polyphonyCache = new polyphonyCacheItem[polyphonyCacheSize];
  // https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
  libremidi::midi_out midi = libremidi::midi_out(libremidi::API::LINUX_ALSA_SEQ, "test");
  midi.open_port(1);
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
  while (true) {
    midiMutex.lock();
    if (false) {
    } else if (mouthOpen && !midiPlaying) {
      channel = 1;
      note = 60 + scaleDegree;
      velocity = round(127 * (1.0 - mouthWidePercentage));
      timePoint = std::chrono::high_resolution_clock::now();
      // FIXME: polyphony
      if (polyphonyCache[polyphonyCacheIndex].note == note) {
        midi.send_message(libremidi::message::note_off(channel, note, velocity));
      }
      polyphonyCache[polyphonyCacheIndex].note = note;
      polyphonyCache[polyphonyCacheIndex].entryTime = timePoint;
      midi.send_message(libremidi::message::note_on(channel, note, velocity));
      midiPlaying = true;
      polyphonyCacheIndex = (polyphonyCacheIndex + 1) % polyphonyCacheSize;
    } else if (!mouthOpen && midiPlaying) {
      channel = 1;
      note = 60 + scaleDegree;
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
    std::thread thread_calibration(calibration);
    std::thread thread_interpretFacialData(interpretFacialData);
    std::thread thread_midiDriver(midiDriver);

    cv::VideoCapture cap(0);
    // FIXME: work to figure out optimal size
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


    image_window win;
    std::vector<rectangle> faces;
    while (!win.is_closed()) {
      m.lock();
      calibrationGreenLight = true;
      m.unlock();
      cv::Mat matrix;
      if (!cap.read(matrix)) {
        break;
      }
      // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
      // wraps the Mat object, it doesn't copy anything.  So baseimg is only valid as
      // long as matrix is valid.  Also don't do anything to matrix that would cause it
      // to reallocate the memory which stores the image as that will make baseimg
      // contain dangling pointers.  This basically means you shouldn't modify matrix
      // while using baseimg.
      cv::Mat guiMatrix = cv::Mat(matrix.size(), matrix.type(), cv::Scalar::all(255));
      cv::flip(guiMatrix, guiMatrix, 1);
      // set color to grayscale
      cv::cvtColor(matrix, matrix, cv::COLOR_BGR2GRAY);
      cv::flip(matrix, matrix, 1);
      // use unsigned char instead of `bgr_pixel` as pixel type,
      // because we are working with grayscale magnitude
      cv_image<unsigned char> baseimg(matrix);
      // detect faces
      faces = detector(baseimg);
      // find the pose of each face
      // assume only one face is detected, because only one person will be using this
      if (!faces.empty()) {
        full_object_detection shape = sp(baseimg, faces[0]);

        // NOTE: for testing only
        if (false) {
          eyeTracking(&win, &matrix, &shape);
        }

        m.lock();
        // draw a pointer directed by the nose
        if (noseZeroCalibrationComplete) {
          cv::circle(guiMatrix, cv::Point(noseCurrentPositionMultiplied[0], noseCurrentPositionMultiplied[1]), 1, cv::Scalar(0, 0, 255), 1);
        } else {
          cv::circle(guiMatrix, cv::Point(noseCurrentPosition[0], noseCurrentPosition[1]), 1, cv::Scalar(0, 0, 255), 1);
        }
        // draw out pitch board when all necessary parameters are configured
        if (true
          && nosePitchBoardUpperLeftComplete
          && nosePitchBoardUpperRightComplete
          && nosePitchBoardLowerRightComplete
          && nosePitchBoardLowerLeftComplete) {
          if (nosePitchBoardCalculateComplete) {
            // connect all outer points
            for (int i = 0; i <= pitchBoardNumberOfFrets; i++) {
              cv::line(guiMatrix,
                  cv::Point(pitchBoardPoints[i][0][0], pitchBoardPoints[i][0][1]),
                  cv::Point(pitchBoardPoints[i][pitchBoardNumberOfStrings][0], pitchBoardPoints[i][pitchBoardNumberOfStrings][1]),
                  cv::Scalar(0, 0, 0));
            }
            for (int j = 0; j <= pitchBoardNumberOfStrings; j++) {
              cv::line(guiMatrix,
                  cv::Point(pitchBoardPoints[0][j][0], pitchBoardPoints[0][j][1]),
                  cv::Point(pitchBoardPoints[pitchBoardNumberOfFrets][j][0], pitchBoardPoints[pitchBoardNumberOfFrets][j][1]),
                  cv::Scalar(0, 0, 0));
            }
          }
        } else {
          // draw specified corners of the pitch board because it's not all mapped out yet
          if (nosePitchBoardUpperLeftComplete) {
            cv::circle(guiMatrix, cv::Point(nosePitchBoardUpperLeftPosition[0], nosePitchBoardUpperLeftPosition[1]), 1, cv::Scalar(0, 0, 0), 2);
          }
          if (nosePitchBoardUpperRightComplete) {
            cv::circle(guiMatrix, cv::Point(nosePitchBoardUpperRightPosition[0], nosePitchBoardUpperRightPosition[1]), 1, cv::Scalar(0, 0, 0), 2);
          }
          if (nosePitchBoardLowerRightComplete) {
            cv::circle(guiMatrix, cv::Point(nosePitchBoardLowerRightPosition[0], nosePitchBoardLowerRightPosition[1]), 1, cv::Scalar(0, 0, 0), 2);
          }
          if (nosePitchBoardLowerLeftComplete) {
            cv::circle(guiMatrix, cv::Point(nosePitchBoardLowerLeftPosition[0], nosePitchBoardLowerLeftPosition[1]), 1, cv::Scalar(0, 0, 0), 2);
          }
        }
        m.unlock();
        m.lock();
        int guiMatrixWidth = guiMatrix.cols;
        int guiMatrixHeight = guiMatrix.rows;
        cv::line(guiMatrix,
            cv::Point(guiMatrixWidth - 1, round(guiMatrixHeight / 2) - round(mouthOpenPercentage * (float)guiMatrixHeight / 4.0)),
            cv::Point(guiMatrixWidth - 1 - 5, round(guiMatrixHeight / 2) - round(mouthOpenPercentage * (float)guiMatrixHeight / 4.0)),
            cv::Scalar(0, 0, 0),
            2);
        cv::line(guiMatrix,
            cv::Point(guiMatrixWidth - 1, round(capPropFrameHeight / 2) + round(mouthOpenPercentage * (float)guiMatrixHeight / 4.0)),
            cv::Point(guiMatrixWidth - 1 - 5, round(capPropFrameHeight / 2) + round(mouthOpenPercentage * (float)guiMatrixHeight / 4.0)),
            cv::Scalar(0, 0, 0),
            2);
        cv::line(guiMatrix, 
            cv::Point(round(guiMatrixWidth / 2) - round(mouthWidePercentage * (float)guiMatrixWidth / 4.0), guiMatrixHeight - 1),
            cv::Point(round(guiMatrixWidth / 2) - round(mouthWidePercentage * (float)guiMatrixWidth / 4.0), guiMatrixHeight - 1 - 5),
            cv::Scalar(0, 0, 0),
            2);
        cv::line(guiMatrix, 
            cv::Point(round(guiMatrixWidth / 2) + round(mouthWidePercentage * (float)guiMatrixWidth / 4.0), guiMatrixHeight - 1),
            cv::Point(round(guiMatrixWidth / 2) + round(mouthWidePercentage * (float)guiMatrixWidth / 4.0), guiMatrixHeight - 1 - 5),
            cv::Scalar(0, 0, 0),
            2);
        m.unlock();

        // capture nose position
        m.lock();
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
        
        cv_image<bgr_pixel> guiImg(guiMatrix);
        win.clear_overlay();
        if (true
          && noseZeroCalibrationComplete
          && mouthCalibrationClosedComplete
          && mouthCalibrationOpenUpDownComplete
          && mouthCalibrationOpenRightLeftComplete) {
          win.set_image(guiImg);
        } else {
          win.set_image(baseimg);
          win.add_overlay(render_face_detections(shape));
        }
      } else {
        win.clear_overlay();
        win.set_image(baseimg);
      }
    }
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
