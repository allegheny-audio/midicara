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

using namespace dlib;
using namespace std;

#if DEBUG
static const bool isDebug = true;
#else
static const bool isDebug = false;
#endif

// for debugging purposes: show an opencv matrix on a display window
void opencvShowGrayscaleMatrix(dlib::image_window* win, cv::Mat* mat) {
  cv_image<unsigned char> myImage(*mat);
  win->clear_overlay();
  win->set_image(myImage);
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


        // mask used to block out area where eyes are located
        cv::Mat eyeMaskMatrix = cv::Mat::zeros(matrix.size(), matrix.type());
        std::vector<cv::Point> leftEyePoints = {
          cv::Point(shape.part(36).x(), shape.part(36).y()),
          cv::Point(shape.part(37).x(), shape.part(37).y()),
          cv::Point(shape.part(38).x(), shape.part(38).y()),
          cv::Point(shape.part(39).x(), shape.part(39).y()),
          cv::Point(shape.part(40).x(), shape.part(40).y()),
          cv::Point(shape.part(41).x(), shape.part(41).y()),
        };
        std::vector<cv::Point> rightEyePoints = {
          cv::Point(shape.part(42).x(), shape.part(42).y()),
          cv::Point(shape.part(43).x(), shape.part(43).y()),
          cv::Point(shape.part(44).x(), shape.part(44).y()),
          cv::Point(shape.part(45).x(), shape.part(45).y()),
          cv::Point(shape.part(46).x(), shape.part(46).y()),
          cv::Point(shape.part(47).x(), shape.part(47).y())
        };
        // use points from dlib's detection to fill in black polygons where eyes are
        cv::fillConvexPoly(eyeMaskMatrix, leftEyePoints, cv::Scalar(255, 255, 255));
        cv::fillConvexPoly(eyeMaskMatrix, rightEyePoints, cv::Scalar(255, 255, 255));
        // 'kernel' used for erosion/dilation algorithm
        cv::Mat kernel = cv::Mat::ones(5, 5, matrix.type());
        // use image dilation to expand the borders of the eye regions
        cv::dilate(eyeMaskMatrix, eyeMaskMatrix, kernel);

        // used to overlay at the end to allow opencv to find contours better within the eye shapes
        cv::Mat eyeBoundaryMatrix = eyeMaskMatrix.clone();
        cv::Mat eyeBoundaryInnerMatrix = eyeMaskMatrix.clone();
        cv::erode(eyeBoundaryInnerMatrix, eyeBoundaryInnerMatrix, cv::Mat());
        cv::bitwise_xor(eyeBoundaryMatrix, eyeBoundaryInnerMatrix, eyeBoundaryMatrix, eyeMaskMatrix);
        cv::bitwise_not(eyeBoundaryMatrix, eyeBoundaryMatrix);

        // matrix of actual eyes in image
        cv::Mat eyesMatrix = cv::Mat::zeros(matrix.size(), matrix.type());
        // get a cropping of just the eyes using the masking and output it to eyeMatrix
        cv::bitwise_and(matrix, matrix, eyesMatrix, eyeMaskMatrix);
        // find mean of values in BW eyesMatrix
        cv::Scalar eyesMatrixMean = cv::mean(eyesMatrix, eyeMaskMatrix);
        // use threshold value to be a function of eyesMatrixMean in order to make it responsive to lighting changes
        unsigned char threshold = floor(eyesMatrixMean[0]) - 10;
        cv::threshold(eyesMatrix, eyesMatrix, threshold, 255, cv::THRESH_BINARY_INV);
        cv::Mat pupilKernel = cv::Mat::ones(3, 3, matrix.type());
        cv::erode(eyesMatrix, eyesMatrix, pupilKernel); // shave off imperfections
        // cv::dilate(eyesMatrix, eyesMatrix, cv::Mat()); // blow up smoother shapes

        cv::bitwise_and(eyeBoundaryMatrix, eyesMatrix, eyesMatrix);
        // FIXME: cv::findContours(eyesMatrix, contours, 30, 255, cv::THRESH_BINARY_INV);
        // { deletable
        // opencvShowGrayscaleMatrix(&win, &eyesMatrix);
        // } deletable
        
        // NOTE: there are always 68 parts
        // FIXME: uncomment: win.clear_overlay();
        if (isDebug) {
          cout << "[LOG] win.clear_overlay()" << endl;
        }
        // FIXME: uncomment: win.set_image(baseimg);
        if (isDebug) {
          cout << "[LOG] win.set_image(baseimg)" << endl;
        }
        // FIXME: uncomment: win.add_overlay(render_face_detections(shape));
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
