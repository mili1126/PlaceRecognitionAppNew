package com.mili.placerecognitionapp.filters;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.graphics.PointF;
import android.support.annotation.NonNull;
import android.util.Log;
import android.widget.Toast;

import com.mili.placerecognitionapp.MainActivity;
import com.mili.placerecognitionapp.R;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.*;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

/**
 * Created by mili on 5/5/16.
 */
public class RecognitionFilter implements Filter {
    private final static String TAG = "RecognitionFilter";

    private List<String> DESCRIPTOR_FOLDERS = Arrays.asList(
            "sift",
            "surf",
            "orb"
    );

    private int mFeatureMode;

    // The reference imgaes;
    private List<Mat> mReferenceImgages = new ArrayList<>();
    // The reference image (this detector's target).
    private Mat mReferenceImage;
    // Descriptors of the reference image's features.
    private List<Mat> mReferenceDescriptors = new ArrayList<>();


    // Features of the scene (the current frame).
    private  MatOfKeyPoint mSceneKeypoints = new MatOfKeyPoint();
    // Descriptors of the scene's features.
    private Mat mSceneDescriptor = new Mat();

    // Tentative matches of scene features and reference features.
    private MatOfDMatch mMatches = new MatOfDMatch();

    // A feature detector, which finds features in images.
    public FeatureDetector mFeatureDetector;
    // A descriptor extractor, which creates descriptors of features.
    public DescriptorExtractor mDescriptorExtractor;
    // A descriptor matcher, which matches features based on their descriptors.
    public DescriptorMatcher mDescriptorMatcher;

    public RecognitionFilter( Context context, int featureMode) throws IOException {
        mFeatureMode = featureMode;
        // Load the reference image from the app's resources.
        // It is loaded in BGR (blue, green, red) format.
        mReferenceImgages.add(Utils.loadResource(context, R.drawable.frame1, Imgcodecs.CV_LOAD_IMAGE_COLOR));
        mReferenceImgages.add(Utils.loadResource(context, R.drawable.frame2, Imgcodecs.CV_LOAD_IMAGE_COLOR));
        mReferenceImgages.add(Utils.loadResource(context, R.drawable.frame3, Imgcodecs.CV_LOAD_IMAGE_COLOR));
        mReferenceImgages.add(Utils.loadResource(context, R.drawable.frame4, Imgcodecs.CV_LOAD_IMAGE_COLOR));
        mReferenceImgages.add(Utils.loadResource(context, R.drawable.frame5, Imgcodecs.CV_LOAD_IMAGE_COLOR));
        mReferenceImgages.add(Utils.loadResource(context, R.drawable.frame6, Imgcodecs.CV_LOAD_IMAGE_COLOR));
        mReferenceImgages.add(Utils.loadResource(context, R.drawable.frame7, Imgcodecs.CV_LOAD_IMAGE_COLOR));
        mReferenceImgages.add(Utils.loadResource(context, R.drawable.frame8, Imgcodecs.CV_LOAD_IMAGE_COLOR));
        mReferenceImgages.add(Utils.loadResource(context, R.drawable.frame9, Imgcodecs.CV_LOAD_IMAGE_COLOR));
        mReferenceImgages.add(Utils.loadResource(context, R.drawable.frame10, Imgcodecs.CV_LOAD_IMAGE_COLOR));

        if (mFeatureMode == 0) {
            //brisk
            mFeatureDetector = FeatureDetector.create(FeatureDetector.ORB);
            mDescriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
            mDescriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);

        } else if (mFeatureMode == 1) {
            //fast
            mFeatureDetector = FeatureDetector.create(FeatureDetector.ORB);
            mDescriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
            mDescriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);

        } else if (mFeatureMode == 2) {
            //orb
            mFeatureDetector = FeatureDetector.create(FeatureDetector.ORB);
            mDescriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
            mDescriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);

        }

        for (int i = 0; i < 10; i++) {
            MatOfKeyPoint mReferenceKeypoints = new MatOfKeyPoint();
            Mat mReferenceDescriptor = new Mat();
            mReferenceImage = mReferenceImgages.get(i);
            Mat referenceImageGray = new Mat();
            Imgproc.cvtColor(mReferenceImage, referenceImageGray,
                    Imgproc.COLOR_BGR2GRAY);
            mFeatureDetector.detect(referenceImageGray, mReferenceKeypoints);
            mDescriptorExtractor.compute(referenceImageGray, mReferenceKeypoints,
                    mReferenceDescriptor);
            mReferenceDescriptors.add(mReferenceDescriptor);
            Log.d(TAG, String.valueOf(mReferenceDescriptor.rows()));
        }
        Log.d(TAG, "Reference descriptors loaded.");
    }

    @Override
    public int apply(Mat src, Mat dst) {
        // Detect the scene features, compute their descriptors,
        // and match the scene descriptors to reference descriptors.
        // Convert the scene to GRAY.
        Mat mGraySrc = new Mat();
        Imgproc.cvtColor(src, mGraySrc, Imgproc.COLOR_RGBA2GRAY);

        mFeatureDetector.detect(mGraySrc, mSceneKeypoints);
        mDescriptorExtractor.compute(mGraySrc, mSceneKeypoints,
                mSceneDescriptor);
        Features2d.drawKeypoints(mGraySrc, mSceneKeypoints, dst);

        int matchIndex = -1;
        int matchSize = 0;
        for (int i = 0; i < 10 ; i++) {

            mDescriptorMatcher.match(mSceneDescriptor,
                    mReferenceDescriptors.get(i), mMatches);

            // Calculate the max and min distances between keypoints.
            double maxDist = 0.0;
            double minDist = Double.MAX_VALUE;
            List<DMatch> matchesList = mMatches.toList();
            for (org.opencv.core.DMatch match : matchesList) {
                double dist = match.distance;
                if (dist < minDist) {
                    minDist = dist;
                }
                if (dist > maxDist) {
                    maxDist = dist;
                }
            }

            Log.d(TAG, "maxDist=" + maxDist + " minDist=" + minDist);

            if (minDist > 50.0) {
                // The target is completely lost.
                return -1 ;
            }

            // Identify "good" keypoints based on match distance.
            int goodNum = 0;
            double maxGoodMatchDist = 0.0;
            if (mFeatureMode == 0) {
                maxGoodMatchDist = Math.max(maxDist/3.0, 2.0*minDist) ;
            } else if (mFeatureMode == 1) {
                maxGoodMatchDist = Math.max(maxDist/3.0, 2.0*minDist) ;
            } else if (mFeatureMode == 2) {
                maxGoodMatchDist = Math.max(maxDist/2.0, 2.0*minDist) ;
            }


            for (final DMatch match : matchesList) {
                if (match.distance < maxGoodMatchDist) {
                    goodNum ++;
                }
            }


            if ( goodNum > matchSize) {
                matchIndex = i;
                matchSize = goodNum;
            }

        }


        return matchIndex;
    }

}
