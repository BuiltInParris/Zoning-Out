#import "MasterViewController.h"
#import "AppDelegate.h"
#import <AudioToolbox/AudioToolbox.h>
#import <AudioToolbox/AudioServices.h>


#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

@interface MasterViewController () {
    NSMutableArray *_objects;
}
@property (weak, nonatomic) IBOutlet UISlider *timeSlider;

@end

@implementation MasterViewController
/*
 - (void)awakeFromNib
 {
 
 if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPad) {
 self.clearsSelectionOnViewWillAppear = NO;
 self.preferredContentSize = CGSizeMake(320.0, 600.0);
 }
 [super awakeFromNib];
 }*/


/** Function Headers */
void detectAndDisplay( cv::Mat frame, MasterViewController*test );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
NSString *path = [[[NSBundle mainBundle] bundlePath] stringByAppendingString:@"/haarcascade_frontalface_alt.xml"];
cv::String face_cascade_name = *new std::string([path UTF8String]);
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

bool firstTime = true;

int eyeLeftXLocations = 0;
int eyeLeftYLocations = 0;

int eyeRightXLocations = 0;
int eyeRightYLocations = 0;

int faceVisible = false;

int stareCount = 0;

- (void)viewDidLoad
{
    [super viewDidLoad];
    
    
    
    /*UIImage *image = [self UIImageFromCVMat:frame];
    UIImageView *imageView = [[UIImageView alloc] initWithImage:image];
    [imageView setTintColor:[UIColor redColor]];
    [self.view addSubview: imageView];*/
    
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        while(true){
            if(stareCount > 10)
            {
                NSLog(@"STOP STARING YOU CREEPER");
                AudioServicesPlayAlertSound(kSystemSoundID_Vibrate);
                stareCount = 0;
            }
        }
    });
    
    dispatch_async(dispatch_get_main_queue(), ^{
        
    cv::VideoCapture capture;
    cv::Mat frame;
    
    // Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); NSLog(@"NAH");};
    
     /*cv::namedWindow(main_window_name,cv::WINDOW_NORMAL);
     cv::moveWindow(main_window_name, 400, 100);
     cv::namedWindow(face_window_name,cv::WINDOW_NORMAL);
     cv::moveWindow(face_window_name, 10, 100);
     cv::namedWindow("Right Eye",cv::WINDOW_NORMAL);
     cv::moveWindow("Right Eye", 10, 600);
     cv::namedWindow("Left Eye",cv::WINDOW_NORMAL);
     cv::moveWindow("Left Eye", 10, 800);
     cv::namedWindow("aa",cv::WINDOW_NORMAL);
     cv::moveWindow("aa", 10, 800);
     cv::namedWindow("aaa",cv::WINDOW_NORMAL);
     cv::moveWindow("aaa", 10, 800);
     */
    createCornerKernels();
    ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
            43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);
                /*if(faceVisible)
                {
                    [self performSelectorOnMainThread:@selector(MyMethodName) withObject:nil waitUntilDone:NO];
                } else {
                    
                }*/
        
    // Read the video stream
    capture = cv::VideoCapture( 1 );
    if(capture.isOpened()) {
        while( true ) {
            capture.read(frame);
            // mirror it
            cv::flip(frame, frame, 1);
            frame.copyTo(debugImage);
            
            // Apply the classifier to the frame
            if( !frame.empty() ) {
                detectAndDisplay( frame, self );
            }
            else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }
            
            //UIImage *image = [self UIImageFromCVMat:frame];
            //UIImageView *imageView = [[UIImageView alloc] initWithFrame:];
            //[imageView setTintColor:[UIColor redColor]];
            //imshow(main_window_name,debugImage);
            //if (cv::waitKey(30) >= 0) break;
            
            //int c = cv::waitKey(10);
             //if( (char)c == 'c' ) { break; }
             //if( (char)c == 'f' ) {
             /*    UIImage *image = [self UIImageFromCVMat:frame];
                 UIImageView *imageView = [[UIImageView alloc] initWithImage:image];
                 [imageView setTintColor:[UIColor redColor]];
                 [self.view addSubview: imageView];*/
             //imwrite("frame.png",frame);
             //}
            
        }
    }
    
    releaseCornerKernels();
    });
    
    
}

cv::Mat findEyes(cv::Mat frame_gray, cv::Rect face) {
    cv::Mat faceROI = frame_gray(face);
    cv::Mat debugFace = faceROI;
    
    if (kSmoothFaceImage) {
        double sigma = kSmoothFaceFactor * face.width;
        GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
    }
    //-- Find eye regions and draw them
    int eye_region_width = face.width * (kEyePercentWidth/100.0);
    int eye_region_height = face.width * (kEyePercentHeight/100.0);
    int eye_region_top = face.height * (kEyePercentTop/100.0);
    cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                           eye_region_top,eye_region_width,eye_region_height);
    cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                            eye_region_top,eye_region_width,eye_region_height);
    
    //-- Find Eye Centers
    cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
    cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
    
    int FacialDiff = 5;
    
    if ((leftPupil.x - eyeLeftXLocations < FacialDiff && leftPupil.x - eyeLeftXLocations > -FacialDiff) &&
        (leftPupil.y - eyeLeftYLocations < FacialDiff && leftPupil.y - eyeLeftYLocations > -FacialDiff) &&
        (rightPupil.x - eyeRightXLocations < FacialDiff && rightPupil.x - eyeRightXLocations > -FacialDiff) &&
        (rightPupil.x - eyeRightXLocations < FacialDiff && rightPupil.x - eyeRightXLocations > -FacialDiff)
        
        )
    {
        stareCount++;
    } else{
        stareCount = 0;
    }
    
    eyeLeftXLocations = leftPupil.x;
    eyeLeftYLocations = leftPupil.y;
    eyeRightXLocations = rightPupil.x;
    eyeRightYLocations = rightPupil.y;
    
    // get corner regions
    /*cv::Rect leftRightCornerRegion(leftEyeRegion);
    leftRightCornerRegion.width -= leftPupil.x;
    leftRightCornerRegion.x += leftPupil.x;
    leftRightCornerRegion.height /= 2;
    leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
    cv::Rect leftLeftCornerRegion(leftEyeRegion);
    leftLeftCornerRegion.width = leftPupil.x;
    leftLeftCornerRegion.height /= 2;
    leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
    cv::Rect rightLeftCornerRegion(rightEyeRegion);
    rightLeftCornerRegion.width = rightPupil.x;
    rightLeftCornerRegion.height /= 2;
    rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
    cv::Rect rightRightCornerRegion(rightEyeRegion);
    rightRightCornerRegion.width -= rightPupil.x;
    rightRightCornerRegion.x += rightPupil.x;
    rightRightCornerRegion.height /= 2;
    rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
    rectangle(debugFace,leftRightCornerRegion,200);
    rectangle(debugFace,leftLeftCornerRegion,200);
    rectangle(debugFace,rightLeftCornerRegion,200);
    rectangle(debugFace,rightRightCornerRegion,200);
    // change eye centers to face coordinates
    rightPupil.x += rightEyeRegion.x;
    rightPupil.y += rightEyeRegion.y;
    leftPupil.x += leftEyeRegion.x;
    leftPupil.y += leftEyeRegion.y;
    // draw eye centers
    circle(debugFace, rightPupil, 3, 1234);
    circle(debugFace, leftPupil, 3, 1234);
    
    //-- Find Eye Corners
    if (kEnableEyeCorner) {
        cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
        leftRightCorner.x += leftRightCornerRegion.x;
        leftRightCorner.y += leftRightCornerRegion.y;
        cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
        leftLeftCorner.x += leftLeftCornerRegion.x;
        leftLeftCorner.y += leftLeftCornerRegion.y;
        cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
        rightLeftCorner.x += rightLeftCornerRegion.x;
        rightLeftCorner.y += rightLeftCornerRegion.y;
        cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
        rightRightCorner.x += rightRightCornerRegion.x;
        rightRightCorner.y += rightRightCornerRegion.y;
        circle(faceROI, leftRightCorner, 3, 200);
        circle(faceROI, leftLeftCorner, 3, 200);
        circle(faceROI, rightLeftCorner, 3, 200);
        circle(faceROI, rightRightCorner, 3, 200);
    }
    */
    return faceROI;
    //imshow(face_window_name, faceROI);
    //[self UIImageFromCVMat:faceROI];
    
    //  cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
    //  cv::Mat destinationROI = debugImage( roi );
    //  faceROI.copyTo( destinationROI );
}


cv::Mat findSkin (cv::Mat &frame) {
    cv::Mat input;
    cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);
    
    cvtColor(frame, input, cv::COLOR_RGB2GRAY);
    
    for (int y = 0; y < input.rows; ++y) {
        const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
        //    uchar *Or = output.ptr<uchar>(y);
        cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input.cols; ++x) {
            cv::Vec3b ycrcb = Mr[x];
            //      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
            if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
                Or[x] = cv::Vec3b(0,0,0);
            }
        }
    }
    return output;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( cv::Mat frame, MasterViewController*test) {
    std::vector<cv::Rect> faces;
    //cv::Mat frame_gray;
    
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(frame, rgbChannels);
    cv::Mat frame_gray = rgbChannels[2];
    
    //cvtColor( frame, frame_gray, CV_BGR2GRAY );
    //equalizeHist( frame_gray, frame_gray );
    //cv::pow(frame_gray, CV_64F, frame_gray);
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE|cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
    //  findSkin(debugImage);
    
    for( int i = 0; i < faces.size(); i++ )
    {
        rectangle(debugImage, faces[i], 1234);
    }
    //-- Show what you got
    if (faces.size() > 0) {
        if(!faceVisible)
            {
            NSLog(@"HAI");
                
                faceVisible = true;
            }
        
            [test makemyownview];
        //NSLog(@"FACE FOUND");
        //cv::Mat view =
        findEyes(frame_gray, faces[0]);
        /*UIImage *image = [test UIImageFromCVMat:view];
        UIImageView *imageView = [[UIImageView alloc] initWithImage:image];
        [test.view addSubview: imageView];*/
    } else {
            [test makemyownview];
            if(faceVisible)
            {
                faceVisible = false;
            }
    }
}

-(void) makemyownview{
    UIView *view = [[UIView alloc]initWithFrame:CGRectMake(0, 0, 200, 200)];
    view.backgroundColor = [UIColor grayColor];
    [self.view addSubview: view];
}



-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}


//return 0;
// Do any additional setup after loading the view, typically from a nib.
//self.navigationItem.leftBarButtonItem = self.editButtonItem;

//UIBarButtonItem *addButton = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemAdd target:self action:@selector(insertNewObject:)];
//self.navigationItem.rightBarButtonItem = addButton;
/*self.detailViewController = (DetailViewController *)[[self.splitViewController.viewControllers lastObject] topViewController];
 
 if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPad)
 {
 AppDelegate * appDel = [UIApplication sharedApplication].delegate;
 
 SampleFacade * sample = appDel->allSamples[0];
 
 DetailViewController * detailController = self.detailViewController;
 
 [detailController setDetailItem:sample];
 [detailController configureView];
 }*/
/*
 - (void)viewDidUnload
 {
 [super viewDidUnload];
 // Release any retained subviews of the main view.
 }
 
 - (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
 {
 if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPhone) {
 return (interfaceOrientation != UIInterfaceOrientationPortraitUpsideDown);
 } else {
 return YES;
 }
 }
 
 
 #pragma mark - Table View
 
 - (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView
 {
 return 1;
 }
 
 - (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section
 {
 AppDelegate * appDel = [UIApplication sharedApplication].delegate;
 return appDel->allSamples.size();
 }
 
 - (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath
 {
 AppDelegate * appDel = [UIApplication sharedApplication].delegate;
 SampleFacade * sample = appDel->allSamples[indexPath.row];
 
 UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"Cell"];
 
 cell.textLabel.text = [sample title];
 cell.imageView.image = [sample smallIcon];
 
 
 return cell;
 }
 
 - (BOOL)tableView:(UITableView *)tableView canEditRowAtIndexPath:(NSIndexPath *)indexPath
 {
 // Return NO if you do not want the specified item to be editable.
 return NO;
 }
 
 - (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath
 {
 if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPad)
 {
 AppDelegate * appDel = [UIApplication sharedApplication].delegate;
 SampleFacade * sample = appDel->allSamples[indexPath.row];
 
 DetailViewController * detailController = self.detailViewController;
 
 [detailController setDetailItem:sample];
 [detailController configureView];
 }
 }
 
 - (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender
 {
 if ([[segue identifier] isEqualToString:@"showDetail"])
 {
 NSIndexPath *indexPath = [self.tableView indexPathForSelectedRow];
 
 AppDelegate * appDel = [UIApplication sharedApplication].delegate;
 SampleFacade * sample = appDel->allSamples[indexPath.row];
 
 [[segue destinationViewController] setDetailItem:sample];
 }
 }*/

@end
