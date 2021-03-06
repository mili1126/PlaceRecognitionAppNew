package com.mili.placerecognitionapp;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.PorterDuff;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.v7.app.ActionBarActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SubMenu;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import com.mili.placerecognitionapp.filters.Filter;
import com.mili.placerecognitionapp.filters.RecognitionFilter;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;


import uk.co.senab.photoview.PhotoViewAttacher;


public class CameraActivity extends ActionBarActivity
        implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "CameraActivity";


    Canvas mCanvas;
    Paint mPaint;
    Drawable imageDrawable;
    Bitmap imageBitmap;
    Bitmap canvasBitmap;
    private FloatingActionButton mButton;
    private PhotoViewAttacher mAttacher;

    private Map<Integer, PointF> routePoint = new HashMap<Integer, PointF>();
    private int matchIndex;


    // Keys for storing.
    private static final String STATE_IMAGE_SIZE_INDEX = "imageSizeIndex";
    private static final String STATE_RECOGNITION_FILTER_INDEX = "recognitionFilterIndex";

    // An ID for items in the image size submenu.
    private static final int MENU_GROUP_ID_SIZE = 1;

    // The camera view.
    private CameraBridgeViewBase mCameraView;

    // The filter
    private Filter[] mRecognitionFilters;
    // The indices of the active filter.
    private int mRecognitionFilterIndex;

    // The image sizes supported by the active camera.
    private List<Camera.Size> mSupportedImageSizes;
    // The index of the active image size.
    private int mImageSizeIndex;

    // Whether an asynchronous menu action is in progress.
    // If so, menu interaction should be disabled.
    private boolean mIsMenuLocked;

    // The OpenCV loader callback.
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(final int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.d(TAG, "OpenCV loaded successfully");
                    mCameraView.enableView();
                    //mCameraView.enableFpsMeter();


                    //create filters
                    final Filter siftFilter;
                    try {
                        siftFilter = new RecognitionFilter(CameraActivity.this, 0);
                    } catch (IOException e) {
                        Log.e(TAG, "Failed to create sift recognition");
                        e.printStackTrace();
                        break;
                    }
                    final Filter surfFilter;
                    try {
                        surfFilter = new RecognitionFilter(CameraActivity.this, 1);
                    } catch (IOException e) {
                        Log.e(TAG, "Failed to create surf recognition");
                        e.printStackTrace();
                        break;
                    }
                    final Filter orbFilter;
                    try {
                        orbFilter = new RecognitionFilter(CameraActivity.this, 2);
                    } catch (IOException e) {
                        Log.e(TAG, "Failed to create orb recognition");
                        e.printStackTrace();
                        break;
                    }

                    mRecognitionFilters = new Filter[]{
                            siftFilter,
                            surfFilter,
                            orbFilter
                    };
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        final Window window = getWindow();
        window.addFlags(
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (savedInstanceState != null) {
            mImageSizeIndex = savedInstanceState.getInt(STATE_IMAGE_SIZE_INDEX, 0);
            mRecognitionFilterIndex = savedInstanceState.getInt(STATE_RECOGNITION_FILTER_INDEX, 0);
        } else {
            mImageSizeIndex = 0;
            mRecognitionFilterIndex = -1;
        }

        final Camera camera;
        camera = Camera.open();
        final Camera.Parameters parameters = camera.getParameters();
        camera.release();
        mSupportedImageSizes = parameters.getSupportedPreviewSizes();
        final Camera.Size size = mSupportedImageSizes.get(mImageSizeIndex);

        //        mCameraView = new JavaCameraView(this, 0);
        mCameraView = (JavaCameraView) findViewById(R.id.camera_view);
//        mCameraView.setMaxFrameSize(size.width, size.height);
        mCameraView.setMaxFrameSize(1280, 768);
        mCameraView.setCvCameraViewListener(this);

        mButton = (FloatingActionButton) findViewById(R.id.image_button);
        mButton.setUseCompatPadding(false);
        mButton.setCompatElevation(.1f);
        mButton.setBackgroundTintMode(null);
        mButton.setTranslationY(-100);
        mAttacher = new PhotoViewAttacher(mButton);

        // Set the Drawable displayed
        imageDrawable = getResources().getDrawable(R.drawable.brown_280_floor_plan);

        imageBitmap = ((BitmapDrawable) imageDrawable).getBitmap();
        canvasBitmap = Bitmap.createBitmap(imageBitmap.getWidth(), imageBitmap.getHeight(), Bitmap.Config.RGB_565);
        mCanvas = new Canvas(canvasBitmap);
        mButton.setImageDrawable(new BitmapDrawable(getResources(), canvasBitmap));
        mCanvas.drawBitmap(imageBitmap, 0, 0, null);

        mPaint = new Paint();
        mPaint.setColor(Color.RED);
        //testing draw circle
        //drawLocation(mCanvas, mPaint, 500.f, 1000.f);

        //read all locations
        readLocation();

        matchIndex = -1;
        new Timer().scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        updateLocation(matchIndex);
                    }
                });

            }
        }, 0, 3000);//put here time 1000 milliseconds=1 second
    }


    @Override
    public boolean onCreateOptionsMenu(final Menu menu) {
        getMenuInflater().inflate(R.menu.activity_camera, menu);

        int numSupportedImageSizes = mSupportedImageSizes.size();
        if (numSupportedImageSizes > 1) {
            final SubMenu sizeSubMenu = menu.addSubMenu(
                    R.string.menu_image_size);
            for (int i = 0; i < numSupportedImageSizes; i++) {
                final Camera.Size size = mSupportedImageSizes.get(i);
                sizeSubMenu.add(MENU_GROUP_ID_SIZE, i, Menu.NONE,
                        String.format("%dx%d", size.width,
                                size.height));
            }
        }
        return true;
    }

    @Override
    public void onSaveInstanceState(Bundle savedInstanceState) {
        // Save the current image size index.
        savedInstanceState.putInt(STATE_IMAGE_SIZE_INDEX, mImageSizeIndex);
        // Save the current recognition filter index.
        savedInstanceState.putInt(STATE_RECOGNITION_FILTER_INDEX, mRecognitionFilterIndex);
        super.onSaveInstanceState(savedInstanceState);
    }

    // Suppress backward incompatibility errors because we provide
    // backward-compatible fallbacks.
    @SuppressLint("NewApi")
    @Override
    public void recreate() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB) {
            super.recreate();
        } else {
            finish();
            startActivity(getIntent());
        }
    }

    @Override
    public void onPause() {
        if (mCameraView != null) {
            mCameraView.disableView();
        }
        super.onPause();
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0,
                this, mLoaderCallback);
        mIsMenuLocked = false;
    }

    @Override
    public void onDestroy() {
        if (mCameraView != null) {
            mCameraView.disableView();
        }
        super.onDestroy();
    }

    // Suppress backward incompatibility errors because we provide
    // backward-compatible fallbacks (for recreate).
    @SuppressLint("NewApi")
    @Override
    public boolean onOptionsItemSelected(final MenuItem item) {
        if (mIsMenuLocked) {
            return true;
        }
        if (item.getGroupId() == MENU_GROUP_ID_SIZE) {
            mImageSizeIndex = item.getItemId();
            recreate();

            return true;
        }
        switch (item.getItemId()) {
            case R.id.menu_stop:
                mRecognitionFilterIndex = -1;
                Log.d(TAG, "Stop clicked");
                return true;
            case R.id.menu_sift:
                mRecognitionFilterIndex = 0;
                Log.d(TAG, "SIFT clicked");
                return true;
            case R.id.menu_surf:
                mRecognitionFilterIndex = 1;
                Log.d(TAG, "SURF clicked");
                return true;
            case R.id.menu_orb:
                Log.d(TAG, "ORB clicked");
                mRecognitionFilterIndex = 2;
                return true;

            default:
                return super.onOptionsItemSelected(item);
        }
    }

    @Override
    public void onCameraViewStarted(final int width,
                                    final int height) {
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(final CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        final Mat rgba = inputFrame.rgba();

        Calendar c = Calendar.getInstance();
        int seconds = c.get(Calendar.SECOND);
        // Apply the active filters.
        if ((seconds % 5 == 0) && mRecognitionFilters != null && mRecognitionFilterIndex >= 0) {
            Log.d(TAG, "check starts..." + mRecognitionFilterIndex);
            matchIndex = mRecognitionFilters[mRecognitionFilterIndex].apply(rgba, rgba);
            Log.d(TAG, "return match = " + matchIndex);
        }

        return rgba;
    }

    @SuppressLint("NewApi")
    public void readLocation() {
        //read office digit number and office x,y coordinate from txt file
        //split[0]: office digit number
        //split[1]: office x coordinate in float
        //split[2]: office y coordinate in float
        try (BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("location.txt"), "UTF-8"))) {
            String line;
            String[] split;

            while ((line = br.readLine()) != null) {
                // process the line.
                split = line.split("\\s+");
                PointF point = new PointF();
                point.x = Float.parseFloat(split[1]);
                point.y = Float.parseFloat(split[2]);
                routePoint.put(Integer.parseInt(split[0]), point);
//                mPaint.setColor(Color.BLUE);
//                drawLocation(mCanvas, mPaint, point.x, point.y); //draw circles on the map as office location
            }
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void drawLocation(Canvas canvas, Paint paint, float x, float y) {
        canvas.drawCircle(x, y, 80f, paint);

    }


    private void updateLocation(int matchedIndex) {
        if (matchedIndex != -1) {
//            Toast.makeText(getApplicationContext(), "Match location " + (matchedIndex+1), Toast.LENGTH_LONG).show();


            Log.d(TAG, "Draw circle at place " + (matchedIndex+1));

            //draw circles on the map as current location

            // Set the Drawable displayed
            imageDrawable = getResources().getDrawable(R.drawable.brown_280_floor_plan);

            imageBitmap = ((BitmapDrawable) imageDrawable).getBitmap();
            canvasBitmap = Bitmap.createBitmap(imageBitmap.getWidth(), imageBitmap.getHeight(), Bitmap.Config.RGB_565);
            mCanvas = new Canvas(canvasBitmap);
            mButton.setImageDrawable(new BitmapDrawable(getResources(), canvasBitmap));
            mCanvas.drawBitmap(imageBitmap, 0, 0, null);

            mPaint = new Paint();
            mPaint.setColor(Color.RED);
            PointF point = routePoint.get(matchIndex);
            drawLocation(mCanvas, mPaint, point.x, point.y);
        }
    }
}
