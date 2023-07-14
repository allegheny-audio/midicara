#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <math.h>
#include <libremidi/libremidi.hpp>

using namespace dlib;
using namespace std;

#if DEBUG
static const bool isDebug = true;
#else
static const bool isDebug = false;
#endif

std::mutex m;

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
    m.lock();
      int noseDeltaX = noseCurrentPosition[0] - noseZeroCalibrated[0];
      int noseDeltaY = noseCurrentPosition[1] - noseZeroCalibrated[1];
      double atanRes = atan2(noseDeltaY, noseDeltaX);
      double magnitude = round(sqrt(noseDeltaX * noseDeltaX + noseDeltaY * noseDeltaY));
      atanRes = atanRes < 0 ? atanRes + 6.28 : atanRes;
      double scaleDegree = round((atanRes) / 6.28 * 12.0);
    m.unlock();

    m.lock();
    // do stuff with mouth dimensions
    if (mouthOuterLipUpDownCurrentDistance > mouthOuterLipUpDownClosedDistanceCalibrated) {
      cout << "mouth open " <<
        ((float)((mouthOuterLipUpDownCurrentDistance - mouthOuterLipUpDownClosedDistanceCalibrated + mouthInnerLipUpDownCurrentDistance - mouthInnerLipUpDownClosedDistanceCalibrated) / 2) / ((mouthOuterLipUpDownOpenDistanceCalibrated - mouthOuterLipUpDownClosedDistanceCalibrated + mouthInnerLipUpDownOpenDistanceCalibrated - mouthInnerLipUpDownClosedDistanceCalibrated) / 2)) << endl;
    } else {
      cout << "mouth closed" << endl;
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
  // FIXME
  libremidi::observer::callbacks midiCallback;
  midiCallback.output_added = [](int index, std::string name) {
    cout << "[libremidiCallback]: output added (index: " << index << ", name: " << name << ")" << endl;
  };
  // https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
  libremidi::midi_out midi = libremidi::midi_out(libremidi::API::LINUX_ALSA_SEQ, "midi client");
  midi.open_port(0);
  while (true) {
    midi.send_message(libremidi::message::note_on((uint8_t)1, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)2, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)3, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)4, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)5, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)6, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)7, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)8, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)9, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)10, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)11, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)12, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)13, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)14, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)15, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_on((uint8_t)16, (uint8_t)48, (uint8_t)127));
    std::this_thread::sleep_for(1000ms);
    midi.send_message(libremidi::message::note_off((uint8_t)1, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)2, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)3, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)4, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)5, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)6, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)7, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)8, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)9, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)10, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)11, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)12, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)13, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)14, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)15, (uint8_t)48, (uint8_t)127));
    midi.send_message(libremidi::message::note_off((uint8_t)16, (uint8_t)48, (uint8_t)127));
    cout << "message apparently sent." << endl;
    cout << "Port open status: " << (midi.is_port_open() ? "opened" : "unopened" ) << endl;;
  }

}

int main(int argc, char** argv) {  
  try {
    std::thread thread_calibration(calibration);
    std::thread thread_interpretFacialData(interpretFacialData);
    std::thread thread_midiDriver(midiDriver);

    cv::VideoCapture cap(0);
    // work to figure out best size
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 70);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 100);
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

        // capture nose position
        m.lock();
        noseCurrentPosition[0] = shape.part(34-1).x();
        noseCurrentPosition[1] = shape.part(34-1).y();
        m.unlock();
        // capture mouth dimensions
        m.lock();
        mouthOuterLipUpDownCurrentDistance = shape.part(58-1).y() - shape.part(52-1).y();
        mouthOuterLipRightLeftCurrentDistance = shape.part(55-1).x() - shape.part(49-1).x();
        m.unlock();
        // opencvShowGrayscaleMatrix(&win, &matrix);
        
        win.clear_overlay();
        if (isDebug) {
          cout << "[LOG] win.clear_overlay()" << endl;
        }
        win.set_image(baseimg);
        if (isDebug) {
          cout << "[LOG] win.set_image(baseimg)" << endl;
        }
        win.add_overlay(render_face_detections(shape));
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
