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

const unsigned int capPropFrameWidth = 70;
const unsigned int capPropFrameHeight = 100;

bool calibrationGreenLight = false;
bool noseCalibrationInPosition = false;
bool noseCalibrationComplete = false;

bool mouthCalibrationClosedInPosition = false;
bool mouthCalibrationClosedComplete = false;
bool mouthCalibrationOpenUpDownInPosition = false;
bool mouthCalibrationOpenUpDownComplete = false;
bool mouthCalibrationOpenRightLeftInPosition = false;
bool mouthCalibrationOpenRightLeftComplete = false;

int noseCurrentPosition[2];
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

bool midiPlaying;
bool mouthOpen;
double mouthOpenPercentage;
double mouthWidePercentage;
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
  // FIXME: cv::findContours(eyesMatrix, contours, 30, 255, cv::THRESH_BINARY_INV);
  // { deletable
  opencvShowGrayscaleMatrix(win, &eyesMatrix);
  // } deletable
}

void interpretFacialData() {
  // wait for calibration before running
  while (true) {
    m.lock();
    if (true
      && noseCalibrationComplete
      && mouthCalibrationClosedComplete
      && mouthCalibrationOpenUpDownComplete
      && mouthCalibrationOpenRightLeftComplete) {
      m.unlock();
      break;
    }
    m.unlock();
    std::this_thread::sleep_for(100ms);
  }
  if (isDebug) {
    cout << "[LOG] interpretFacialData() green light" << endl;
  }
  while (true) {
    // handle nose dimensions
    m.lock();
    int noseDeltaX = noseCurrentPosition[0] - noseZeroCalibrated[0];
    int noseDeltaY = noseCurrentPosition[1] - noseZeroCalibrated[1];
    m.unlock();
    double atanRes = atan2(noseDeltaY, noseDeltaX);
    atanRes = atanRes < 0 ? atanRes + 6.28 : atanRes;
    midiMutex.lock();
    noseDistanceMagnitude = round(sqrt(noseDeltaX * noseDeltaX + noseDeltaY * noseDeltaY));
    scaleDegree = round((atanRes) / 6.28 * 12.0);
    midiMutex.unlock();

    // handle mouth dimensions
    m.lock();
    if (mouthOuterLipUpDownCurrentDistance > mouthOuterLipUpDownClosedDistanceCalibrated) {
      midiMutex.lock();
      mouthOpen = true;
      mouthOpenPercentage = (double)(
          (double)((mouthOuterLipUpDownCurrentDistance - mouthOuterLipUpDownClosedDistanceCalibrated + mouthInnerLipUpDownCurrentDistance - mouthInnerLipUpDownClosedDistanceCalibrated) / 2)
          /
          (double)((mouthOuterLipUpDownOpenDistanceCalibrated - mouthOuterLipUpDownClosedDistanceCalibrated + mouthInnerLipUpDownOpenDistanceCalibrated - mouthInnerLipUpDownClosedDistanceCalibrated) / 2)
      );
      mouthWidePercentage = (double)(
          (double)((mouthOuterLipRightLeftCurrentDistance - mouthOuterLipRightLeftClosedDistanceCalibrated + mouthInnerLipRightLeftCurrentDistance - mouthInnerLipRightLeftClosedDistanceCalibrated) / 2)
          /
          (double)((mouthOuterLipRightLeftOpenDistanceCalibrated - mouthOuterLipRightLeftClosedDistanceCalibrated + mouthInnerLipRightLeftOpenDistanceCalibrated - mouthInnerLipRightLeftClosedDistanceCalibrated) / 2)
      );
      midiMutex.unlock();
    } else {
      midiMutex.lock();
      mouthOpen = false;
      midiMutex.unlock();
    }
    m.unlock();
  }
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
  if (isDebug) {
    cout << "[LOG] calibration() green light" << endl;
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
  noseCalibrationComplete = true;
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
}

void midiDriver() {
  // initialize polyphonyCache variable
  polyphonyCache = new polyphonyCacheItem[polyphonyCacheSize];
  libremidi::observer::callbacks midiCallback;
  midiCallback.output_added = [](int index, std::string name) {
    cout << "[libremidiCallback]: output added (index: " << index << ", name: " << name << ")" << endl;
  };
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
      velocity = round(127 * (mouthWidePercentage > 1.0 ? 0 : 1 - mouthWidePercentage));
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
      if (isDebug) {
        cout << "[LOG] while entered" << endl;
      }
      cv::Mat matrix;
      if (!cap.read(matrix)) {
        break;
      }
      if (isDebug) {
        cout << "[LOG] cap read" << endl;
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
      if (isDebug) {
        cout << "[LOG] baseimg(matrix)" << endl;
      }
      // detect faces
      faces = detector(baseimg);
      if (isDebug) {
        cout << "[LOG] detector(baseimg)" << endl;
      }
      // find the pose of each face
      // assume only one face is detected, because only one person will be using this
      if (!faces.empty()) {
        full_object_detection shape = sp(baseimg, faces[0]);
        if (isDebug) {
          cout << "[LOG] sp(baseimg, faces[0])" << endl;
        }
        if (isDebug) {
          cout << "[LOG] num_parts()" << endl;
        }
        
        // NOTE: for testing only
        if (false) {
          eyeTracking(&win, &matrix, &shape);
        }

        m.lock();
        if (noseCalibrationComplete) {
          for (int i = 0; i < 12; i++) {
            double theta = ((double) i / 12.0) * 6.28 - 6.28 / 24;
            midiMutex.lock();
            cv::line(guiMatrix, cv::Point(noseZeroCalibrated[0], noseZeroCalibrated[1]), cv::Point(round(cos(theta) * (double)noseDistanceMagnitude) + noseZeroCalibrated[0], round(sin(theta) * (double)noseDistanceMagnitude) + noseZeroCalibrated[1]), cv::Scalar(0, 0, 0));
            midiMutex.unlock();
          }
        }
        m.unlock();
        m.lock();
        cv::circle(guiMatrix, cv::Point(noseCurrentPosition[0], noseCurrentPosition[1]), 1, cv::Scalar(0, 0, 255), 2);
        cv::line(guiMatrix,
            cv::Point(capPropFrameWidth - 1, round(capPropFrameHeight / 2) - (mouthOuterLipUpDownCurrentDistance / 2)),
            cv::Point(capPropFrameWidth - 1 - 5, round(capPropFrameHeight / 2) - (mouthOuterLipUpDownCurrentDistance / 2)),
            cv::Scalar(0, 0, 0),
            2);
        cv::line(guiMatrix,
            cv::Point(capPropFrameWidth - 1, round(capPropFrameHeight / 2) + (mouthOuterLipUpDownCurrentDistance / 2)),
            cv::Point(capPropFrameWidth - 1 - 5, round(capPropFrameHeight / 2) + (mouthOuterLipUpDownCurrentDistance / 2)),
            cv::Scalar(0, 0, 0),
            2);
        cv::line(guiMatrix, 
            cv::Point(round(capPropFrameWidth / 2) - (mouthOuterLipRightLeftCurrentDistance / 2), capPropFrameHeight - 1),
            cv::Point(round(capPropFrameWidth / 2) - (mouthOuterLipRightLeftCurrentDistance / 2), capPropFrameHeight - 1 - 5),
            cv::Scalar(0, 0, 0),
            2);
        cv::line(guiMatrix, 
            cv::Point(round(capPropFrameWidth / 2) + (mouthOuterLipRightLeftCurrentDistance / 2), capPropFrameHeight - 1),
            cv::Point(round(capPropFrameWidth / 2) + (mouthOuterLipRightLeftCurrentDistance / 2), capPropFrameHeight - 1 - 5),
            cv::Scalar(0, 0, 0),
            2);
        m.unlock();

        // capture nose position
        m.lock();
        noseCurrentPosition[0] = shape.part(31-1).x();
        noseCurrentPosition[1] = shape.part(31-1).y();
        m.unlock();
        // capture mouth dimensions
        m.lock();
        mouthOuterLipUpDownCurrentDistance = shape.part(58-1).y() - shape.part(52-1).y();
        mouthOuterLipRightLeftCurrentDistance = shape.part(55-1).x() - shape.part(49-1).x();
        m.unlock();
        // opencvShowGrayscaleMatrix(&win, &matrix);
        
        cv_image<bgr_pixel> guiImg(guiMatrix);
        win.clear_overlay();
        if (isDebug) {
          cout << "[LOG] win.clear_overlay()" << endl;
        }
        win.set_image(guiImg);
        if (isDebug) {
          cout << "[LOG] win.set_image(baseimg)" << endl;
        }
        // win.add_overlay(render_face_detections(shape));
        if (isDebug) {
          cout << "[LOG] end of while, going back" << endl;
        }
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
