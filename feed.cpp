// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

  This example program shows how to find frontal human faces in an image and
  estimate their pose.  The pose takes the form of 68 landmarks.  These are
  points on the face such as the corners of the mouth, along the eyebrows, on
  the eyes, and so forth.  
  


  The face detector we use is made using the classic Histogram of Oriented
  Gradients (HOG) feature combined with a linear classifier, an image pyramid,
  and sliding window detection scheme.  The pose estimator was created by
  using dlib's implementation of the paper:
     One Millisecond Face Alignment with an Ensemble of Regression Trees by
     Vahid Kazemi and Josephine Sullivan, CVPR 2014
  and was trained on the iBUG 300-W face landmark dataset (see
  https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
     300 faces In-the-wild challenge: Database and results. 
     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
  You can get the trained model file from:
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
  Note that the license for the iBUG 300-W dataset excludes commercial use.
  So you should contact Imperial College London to find out if it's OK for
  you to use this model file in a commercial product.


  Also, note that you can train your own models using dlib's machine learning
  tools.  See train_shape_predictor_ex.cpp to see an example.

  


  Finally, note that the face detector is fastest when compiled with at least
  SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
  chip then you should enable at least SSE2 instructions.  If you are using
  cmake to compile this program you can enable them by using one of the
  following commands when you create the build project:
    cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
    cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
    cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
  This will set the appropriate compiler options for GCC, clang, Visual
  Studio, or the Intel compiler.  If you are using another compiler then you
  need to consult your compiler's manual to determine how to enable these
  instructions.  Note that AVX is the fastest but requires a CPU from at least
  2011.  SSE4 is the next fastest and is supported by most current machines.  
*/


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

bool noseCalibrationInPosition = false;
bool noseCalibrationComplete = false;

bool mouthCalibrationClosedInPosition = false;
bool mouthCalibrationClosedComplete = false;
bool mouthCalibrationOpenUpDownInPosition = false;
bool mouthCalibrationOpenUpDownComplete = false;
bool mouthCalibrationOpenRightLeftInPosition = false;
bool mouthCalibrationOpenRightLeftComplete = false;

int noseZeroCalibrated[2];
int noseCurrentPosition[2];
int mouthOuterLipUpDownCurrentDistance;
int mouthOuterLipRightLeftCurrentDistance;
int mouthOuterLipUpDownClosedDistanceCalibrated;
int mouthOuterLipRightLeftClosedDistanceCalibrated;
int mouthOuterLipUpDownOpenDistanceCalibrated;
int mouthOuterLipRightLeftOpenDistanceCalibrated;

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

void calibration() {
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
  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#  Saving position...            #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  m.lock();
  noseCalibrationInPosition = true;
  m.unlock();
  while (true) {
    m.lock();
    if (noseCalibrationComplete) {
      cout << "##################################" << endl;
      cout << "#                                #" << endl;
      cout << "#            Done!               #" << endl;
      cout << "#                                #" << endl;
      cout << "##################################" << endl;
      m.unlock();
      break;
    }
    m.unlock();
  }

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
  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#       Saving position...       #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  m.lock();
  mouthCalibrationClosedInPosition = true;
  m.unlock();
  while (true) {
    m.lock();
    if (mouthCalibrationClosedComplete) {
      cout << "##################################" << endl;
      cout << "#                                #" << endl;
      cout << "#            Done!               #" << endl;
      cout << "#                                #" << endl;
      cout << "##################################" << endl;
      m.unlock();
      break;
    }
    m.unlock();
  }

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
  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#       Saving position...       #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  m.lock();
  mouthCalibrationOpenUpDownInPosition = true;
  m.unlock();
  while (true) {
    m.lock();
    if (mouthCalibrationOpenUpDownComplete) {
      cout << "##################################" << endl;
      cout << "#                                #" << endl;
      cout << "#            Done!               #" << endl;
      cout << "#                                #" << endl;
      cout << "##################################" << endl;
      m.unlock();
      break;
    }
    m.unlock();
  }

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
  cout << "##################################" << endl;
  cout << "#                                #" << endl;
  cout << "#       Saving position...       #" << endl;
  cout << "#                                #" << endl;
  cout << "##################################" << endl;
  m.lock();
  mouthCalibrationOpenRightLeftInPosition = true;
  m.unlock();
  while (true) {
    m.lock();
    if (mouthCalibrationOpenRightLeftComplete) {
      cout << "##################################" << endl;
      cout << "#                                #" << endl;
      cout << "#            Done!               #" << endl;
      cout << "#                                #" << endl;
      cout << "##################################" << endl;
      m.unlock();
      break;
    }
    m.unlock();
  }
}

int main(int argc, char** argv) {  
  try {
    cv::VideoCapture cap(0);
    // work to figure out best size
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 70);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 100);
    // FIXME: explore to see if this can be black and white cap.set(cv::CAP_PROP_CONVERT_RGB, 0);
    if (!cap.isOpened()) {
      cerr << "Unable to connect to camera" << endl;
      return 1;
    }
    if (argc == 1) {
      cout << "Call this program like this:" << endl;
      cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat" << endl;
      cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
      cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
      return 0;
    }

    // We need a face detector.  We will use this to get bounding boxes for
    // each face in an image.
    frontal_face_detector detector = get_frontal_face_detector();
    // And we also need a shape_predictor.  This is the tool that will predict face
    // landmark positions given an image and face bounding box.  Here we are just
    // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
    // as a command line argument.
    shape_predictor sp;
    deserialize(argv[1]) >> sp;


    image_window win;
    std::vector<rectangle> faces;
    // FIXME: eventually start midi process as a side
    std::thread thread_calibration(calibration);
    while (!win.is_closed()) {
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

        // FIXME: this needs to be extracted from the main process
        if (m.try_lock()) {
          noseCurrentPosition[0] = shape.part(34-1).x();
          noseCurrentPosition[1] = shape.part(34-1).y();
          if (noseCalibrationInPosition && !noseCalibrationComplete) {
            noseZeroCalibrated[0] = noseCurrentPosition[0];
            noseZeroCalibrated[1] = noseCurrentPosition[1];
            noseCalibrationComplete = true;
          }
          if (noseCalibrationComplete) {
            int noseDeltaX = (noseCurrentPosition[0] - noseZeroCalibrated[0]);
            int noseDeltaY = (noseCurrentPosition[1] - noseZeroCalibrated[1]);
            // noseDeltaY cannot be 0, so we will by default move it over by 1 if it is.
            if (noseDeltaY == 0) {
              noseDeltaY = 1;
            }
            double atanRes = atan2(noseDeltaY, noseDeltaX);
            double magnitude = round(sqrt(noseDeltaX * noseDeltaX + noseDeltaY * noseDeltaY));
            atanRes = atanRes < 0 ? atanRes + 6.28 : atanRes;
            double scaleDegree = round((atanRes) / 6.28 * 12.0);
            cv::line(matrix, cv::Point(noseCurrentPosition[0], noseCurrentPosition[1]), cv::Point(noseZeroCalibrated[0], noseZeroCalibrated[1]), cv::Scalar(0, 0, 0), 1);
          }
          m.unlock();
        }

        // FIXME: this needs to be extracted from the main process
        if (m.try_lock()) {
          mouthOuterLipUpDownCurrentDistance = shape.part(58-1).y() - shape.part(52-1).y();
          mouthOuterLipRightLeftCurrentDistance = shape.part(55-1).x() - shape.part(49-1).x();
          if (mouthCalibrationClosedInPosition && !mouthCalibrationClosedComplete) {
            mouthOuterLipUpDownClosedDistanceCalibrated = mouthOuterLipUpDownCurrentDistance;
            mouthOuterLipRightLeftClosedDistanceCalibrated = mouthOuterLipRightLeftCurrentDistance;
            mouthCalibrationClosedComplete = true;
          }
          if (mouthCalibrationClosedComplete) {
            // do stuff with mouth dimensions
          }
          if (mouthCalibrationOpenUpDownInPosition && !mouthCalibrationOpenUpDownComplete) {
            mouthOuterLipUpDownOpenDistanceCalibrated = mouthOuterLipUpDownCurrentDistance;
            mouthCalibrationOpenUpDownComplete = true;
          }
          if (mouthCalibrationOpenUpDownComplete) {
            // do stuff with mouth dimensions
          }
          if (mouthCalibrationOpenRightLeftInPosition && !mouthCalibrationOpenRightLeftComplete) {
            mouthOuterLipRightLeftOpenDistanceCalibrated = mouthOuterLipRightLeftCurrentDistance;
            mouthCalibrationOpenRightLeftComplete = true;
          }
          if (mouthCalibrationOpenRightLeftComplete) {
            // do stuff with mouth dimensions
          }
          m.unlock();
        }
        // opencvShowGrayscaleMatrix(&win, &matrix);
        
        // https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
        libremidi::midi_out midi;
        midi.open_virtual_port();
        while (true) {
          midi.send_message(libremidi::message::note_on(1, 48, 127));
          std::this_thread::sleep_for(1000ms);
          midi.send_message(libremidi::message::note_off(1, 48, 127));
          cout << "message apparently sent." << endl;
          cout << "Port open status: " << (midi.is_port_open() ? "opened" : "unopened" ) << endl;;
        }

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
