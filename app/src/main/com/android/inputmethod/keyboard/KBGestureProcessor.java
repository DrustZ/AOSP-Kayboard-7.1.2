package com.android.inputmethod.keyboard;

import android.content.res.Resources;
import android.graphics.Point;
import android.util.Log;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

public class KBGestureProcessor {
    final float Epsilon = (float) 1e-3;
    final float Infinite = (float) 1e10;
    private KeyboardActionListener mListener = null;
    public MainKeyboardView KBView = null;

    ArrayList<Points> pts = new ArrayList<Points>();
    //line detection
    float beta = 0; //slope
    float alpha = 0; //intersection
    float rsquare = 0;

    //logic flags
    boolean openCVinited = false;
    boolean lineDetected = false;
    boolean ringmodeEntered = false;
    boolean notRingmode = false; // check if the user doesn't perform a ring gesture
    int ellipseLocated = 0;

    //control vars
    int linedirection = 0; // no 0 left 1 right 2 up 3 down 4
    int curDirection = 0; // -1 counter clock wise 1 clock wise
    float lastAngle = Infinite; //last moved angle, should be smaller than 360
    Points lastAnglePoint = null;
    int change_count = 0; // count for reverse direction change
    int initial_move_count = 0; // count for initial cursor moves

    float accumAngle = 0; //total angle moved
    float velocity = 0;
    int pointsToDetectLine = 4;
    int maxPoints = 15;
    final int width = Resources.getSystem().getDisplayMetrics().widthPixels;

    ArrayList<org.opencv.core.Point> cPoints = new ArrayList<org.opencv.core.Point>();
    org.opencv.core.Point lastcenter = null; //last circle center
    org.opencv.core.Point backupCenter = null; // in case always can't find a good center in the ringmode
    float centerx = Infinite;
    float centery = Infinite;

    //recognize editing mode
    long gestureStartTime = 0;

    //filters
    StreamingMovingAverage xfilter = new StreamingMovingAverage();
    StreamingMovingAverage yfilter = new StreamingMovingAverage();
    StreamingMovingAverage vfilter = new StreamingMovingAverage();

    private class Points {
        public float x;
        public float y;
        public long time;

        public Points(float x, float y, long time){
            this.x = x;
            this.y = y;
            this.time = time;
        }

        public Points(float x, float y){
            this.x = x;
            this.y = y;
            this.time = 0;
        }

        public float distanceTo(Points p){
            return (float)Math.sqrt((x-p.x)*(x-p.x)+(y-p.y)*(y-p.y));
        }
    }

    private class StreamingMovingAverage{
        public int windowsize = 5;
        public ArrayList<Float> values = new ArrayList<Float>();
        public float sum = 0;

        public float process(float v){
            values.add(v);
            sum += v;
            if (values.size() > windowsize){
                float old = values.remove(0);
                sum -= old;
            }
            return sum/values.size();
        }

        public void clear(){
            sum = 0;
            values.clear();
        }
    }

    public void reset(){
        if (!openCVinited) {
            OpenCVLoader.initDebug();
            openCVinited = true;
        }
        KBView.updateCenter(0, 0, 0xFF00FF00);
        lineDetected = false;
        ringmodeEntered = false;
        notRingmode = false;

        ellipseLocated = 0;
        linedirection = 0;
        curDirection = 0;
        backupCenter = null;
        lastcenter = null;
        centerx = Infinite;
        centery = Infinite;

        lastAngle = Infinite;
        lastAnglePoint = null;
        change_count = 0;
        initial_move_count = 0;
        accumAngle = 0;
        velocity = 0;
        pts.clear();
        cPoints.clear();

        xfilter.clear();
        yfilter.clear();
        vfilter.clear();
        mGestureRecognizer.clear();
        if (mListener != null) mListener.enteringRingMode(false);
    }

    public void setKeyboardActionListener(final KeyboardActionListener listener) {
        mListener = listener;
    }

    //linear interpolation
    private void linearRegress(List<Float> xs, List<Float> ys){
        float xmean = 0;
        float ymean = 0;
        float cov = 0;
        float var = 0;
        float sst = 0;
        float sse = 0;
        int size = xs.size();

        for (int i = 0; i < size; ++i){
            xmean += xs.get(i);
            ymean += ys.get(i);
        }
        xmean /= size;
        ymean /= size;

        for (int i = 0; i < size; ++i){
            float xd = xs.get(i) - xmean;
            float yd = ys.get(i) - ymean;
            var += xd*xd;
            cov += xd*yd;
            sst += yd*yd;
        }
        var = Math.max(var, Epsilon);
        beta = cov / var;
        alpha = ymean - beta * xmean;
        sst = Math.max(sst, Epsilon);

        for (int i = 0; i < size; ++i){
            float df = ys.get(i) - (beta * xs.get(i) + alpha);
            sse += df*df;
        }
        rsquare = 1-sse/sst;

        if (Math.abs(beta) < 0.2 && sst/size < 10){
            //horizontal line
            rsquare = 1;
        }
    }

    private boolean detectLine(int howManyLastPoints){
        int size = pts.size();
        if (size < howManyLastPoints){
            howManyLastPoints = size;
        }
        ArrayList<Float> xs = new ArrayList<Float>();
        ArrayList<Float> ys = new ArrayList<Float>();

        for (int i = size-howManyLastPoints; i < size; ++i){
            xs.add(pts.get(i).x);
            ys.add(pts.get(i).y);
        }
        linearRegress(xs, ys);
        float rs1 = rsquare;
//        Log.e("[Log]", "detectLine: r2 "+rs1 + " slope "+beta);
        linearRegress(ys, xs); //do twice to avoid failure on vertical condition
        float rs2 = rsquare;
//        Log.e("[Log]", "detectLine: r2 "+rs2 + " slope "+beta);

        if (Math.abs(rs1) < 0.6 && Math.abs(rs2) < 0.6){
//            Log.e("[Log]", "detectLine: no line "+rs1+" "+rs2 );
            return false;
        }

        int tmpdirection = 0;
        float ydiff = ys.get(ys.size()-1)-ys.get(ys.size()-4);
        float xdiff = xs.get(xs.size()-1)-xs.get(xs.size()-4);
        if (Math.abs(ydiff) > Math.abs(xdiff)){
            //vertical or horizontal
            if (ydiff > 0) {
                tmpdirection = 4; //down
            } else {
                tmpdirection = 3; //up
            }
        } else {
            if (xdiff > 0) {
                tmpdirection = 2;
            } else {
                tmpdirection = 1;
            }
        }
        if (linedirection == 0 ||
                ((linedirection == 1 || linedirection == 2) && (tmpdirection == 1 || tmpdirection == 2)) ||
                ((linedirection == 3 || linedirection == 4) && (tmpdirection == 3 || tmpdirection == 4)) ){
            linedirection = tmpdirection;
            // call move here~
//            Log.e("[Log]", "line detected"+linedirection);
            return true;
        }
//        Log.e("[Log]", "detect line faliled");
        return false;
    }

    //circle angle calculation
    private float getAngle(float x, float y, float centerx, float centery){
        if (Math.abs(x-centerx) < Epsilon){
            if (centery > y) return 0;
            else return 180;
        }
        float angle = (float)(Math.atan((centery-y)/(x-centerx))/Math.PI);
        if (x > centerx) return 90-angle*180;
        else return 270-angle*180;
    }

    //calculate circle center from three points
    private boolean getCircleCenter(Points pt1, Points pt2, Points pt3){
        float x1 = pt1.x;
        float x2 = pt2.x;
        float x3 = pt3.x;
        float y1 = pt1.y;
        float y2 = pt2.y;
        float y3 = pt3.y;

        float d = 2*(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
        if (Math.abs(d) < Epsilon) {
            centerx = Infinite;
            centery = Infinite;
            return false;
        }
        centerx = ((x1*x1+y1*y1)*(y2-y3) + (x2*x2+y2*y2)*(y3-y1) + (x3*x3+y3*y3)*(y1-y2))/d;
        centery = ((x1*x1+y1*y1)*(x3-x2) + (x2*x2+y2*y2)*(x1-x3) + (x3*x3+y3*y3)*(x2-x1))/d;
        float r = (float)Math.sqrt((centerx-x1)*(centerx-x1)+(centery-y1)*(centery-y1));
//        Log.e("log", "x : "+centerx+" y: "+centery );
        centerx = (float)(centerx*0.9 + x3*0.1);
        centery = (float)(centery*0.9 + y3*0.1);
//        Log.e("log", "after x : "+centerx+" y: "+centery );
        if (r > 600 || centerx < 50 || centery < 50){ //too large
            centerx = Infinite;
            centery = Infinite;
            return false;
        }
        return true;
    }

    private float diffTwoAngles(float a, float b){
        float diff = a-b;
        if (a <= 90 && a >= 0 && b <= 360 && b >= 270){
            diff = a+360-b;
        } else if (a <= 360 && a >= 270 && b <= 90 && b >= 0){
            diff = a-360-b;
        }
        return diff;
    }

    //process points in ring
    private void processRingPoints(){
        Points lastpoint = pts.get(maxPoints-1);
        boolean hasCenter = false;
        if (lastcenter == null) {
//            Log.e("[Log]", "processRingPoints: ["+pts.get(0).x + "," + pts.get(0).y + "],["+pts.get(maxPoints/2).x + "," + pts.get(maxPoints/2).y + "],["+lastpoint.x + "," + lastpoint.y +"]");
            hasCenter = getCircleCenter(pts.get(0), pts.get(maxPoints/2), lastpoint);
        }
        if (!hasCenter){
            if (lastcenter == null) {
                xfilter.clear();
                yfilter.clear();
                vfilter.clear();
                return;
            } else {
                centerx = (float)lastcenter.x;
                centery = (float)lastcenter.y;
            }
        } else {
            centerx = xfilter.process(centerx);
            centery = yfilter.process(centery);
            org.opencv.core.Point newcenter = new org.opencv.core.Point(centerx, centery);
            //if the movement is large or not
            float diff = Math.abs(diffTwoAngles(getAngle(lastpoint.x, lastpoint.y, centerx, centery),
                    getAngle(pts.get(maxPoints - 5).x, pts.get(maxPoints - 5).y, centerx, centery)));
            if (diff > 30){
                backupCenter = newcenter;
            }

            if (lastcenter == null){
                lastcenter = newcenter;
                backupCenter = null;
                KBView.updateCenter((int)lastcenter.x, (int)lastcenter.y, 0xFF00FF00);
            }

            else if (lastcenter == null && backupCenter != null){
                lastcenter = backupCenter;
                backupCenter = null;
                KBView.updateCenter((int)lastcenter.x, (int)lastcenter.y, 0xFF00FF00);
            }
        }

        float a = getAngle(lastpoint.x, lastpoint.y, (float)lastcenter.x, (float)lastcenter.y);
        if (lastAngle == Infinite){
//            lastAngle = getAngle(pts.get(0).x, pts.get(0).y, (float)lastcenter.x, (float)lastcenter.y);
//            lastAngle_time = pts.get(0).time;
            lastAnglePoint = pts.get(0);
            lastAngle = getAngle(pts.get(0).x, pts.get(0).y, (float)lastcenter.x, (float)lastcenter.y);
        }
        float diff = diffTwoAngles(a, lastAngle);

        float threshold = 55;
        float dist = lastAnglePoint.distanceTo(lastpoint);
        if (Math.abs(diff) > threshold && dist > 70){
//            float velocity = 100*Math.abs(diff)/(lastpoint.time-lastAnglePoint.time);
            float velocity = dist/(lastpoint.time-lastAnglePoint.time);
//            Log.e("[Log]", "processRingPoints: velocity : "+velocity );
//            Log.e("[Log]", "dist: "+dist );
            velocity = vfilter.process(velocity);
            initial_move_count += 1;
            if (initial_move_count < 5) {
                velocity = 0;
                if (initial_move_count == 1 && mListener!= null) {mListener.logGesture("ring_gesture");}
            }
            if ((diff < 0 && curDirection == 1) || (diff > 0 && curDirection == -1)){
                if (Math.abs(diff) < 20+threshold){
                    change_count += 1;
//                Log.e("[Diff]", "move diff"+diff+ " angle now "+a);
                    if (change_count >= 1){
                        //lets change direction
                        change_count = 0;
                        curDirection = -curDirection;
                        //move opposite
//                    Log.e("[Log]", "Direction change!" );
                        moveCursor(diff, velocity);
                        cPoints.clear();
                    }
                } else return;
            } else {
                change_count = 0;
                //moving in current direction
                if (diff > 0) curDirection = 1;
                else curDirection = -1;
                moveCursor(diff, velocity);
            }
            accumAngle += diff;
            lastAngle = a;
            lastAnglePoint = lastpoint;
        }

//        if (ellipseLocated <= 1 && Math.abs(accumAngle) > 180 && cPoints.size() > 5){
//            if (locateEllipse())
//                ellipseLocated = 2;
//        }

        if (Math.abs(accumAngle) >= 270){
            lastcenter = null;
            accumAngle = 0;
            if (cPoints.size() > 5){
                if (locateEllipse()) {
                    cPoints.clear();
                    ellipseLocated = 2;
                }
            }
        }
    }

    boolean locateEllipse(){
        MatOfPoint2f matpts = new MatOfPoint2f();
        matpts.fromList(cPoints);
        RotatedRect elipse = Imgproc.fitEllipse(matpts);
        if (elipse.center.x < 50 || elipse.center.y < 50 || elipse.center.y > 550)
            return false;
        lastcenter = elipse.center;
        KBView.updateCenter((int)lastcenter.x, (int)lastcenter.y, 0xFF00FFFF);
        return true;
    }

    //the main logic: get points and process
    public void processGestureEvent(int x, int y, long eventime){
        int size = pts.size();
        if (size == 0) {
            pts.add(new Points(x, y, eventime));
            gestureStartTime = eventime;
            return;
        }

        //for gesture recognizer
        if (eventime - gestureStartTime < 500){
            mGestureRecognizer.addPoint(x, y);
        }

        //too short
        if (eventime-pts.get(size-1).time < 20) return;
        pts.add(new Points(x, y, eventime));
        if (pts.size() > maxPoints) pts.remove(0);

        int threshold = 200;
        if (mListener != null && mListener.isInEditingMode()){
            threshold = 500;
        }
        if (eventime - gestureStartTime > threshold) {
            if (!ringmodeEntered) {
                ringmodeEntered = true;//no longer line, entering ring mode
//                mListener.kbVibrate();
                if (mListener != null) { mListener.enteringRingMode(true); }
            }
            if (pts.size() >= maxPoints){
                cPoints.add(new org.opencv.core.Point(x, y));
                //other gestures have 500 ms to accomplish
                processRingPoints();
            }
        }
    }

    private void moveCursor(float diff, float velocity){
        if (mListener != null){

            int movedirection = 0;

            if (curDirection == 1 ){
                // clockwise: right
                movedirection = 2;
            } else {
                movedirection = 1;
            }
            //CD-gain
            int movetime = 1;
            if (curDirection <= 2 && velocity > 1.7){
//                movetime = (int)Math.ceil(5*(1-Math.exp(0.04*(35-velocity))));
                movetime = 2;
            }
            if (velocity < 25){
                mListener.kbVibrate();
            }
            for (int i = 0; i < movetime; i++) {
                mListener.moveCursor(movedirection, false);
            }
        }
    }

    //editing gestures

    public void fingerLifted(long eventime){
        if (eventime - gestureStartTime < 500){
            //begin to recognize editing gestures
            String res = mGestureRecognizer.recognize();
            switch (res){
                case "copy":
                    mListener.copyText();
                    break;
                case "paste":
                    mListener.pasteText();
                    break;
                case "cut":
                    mListener.cutText();
                    break;
                case "undo":
                    mListener.undoEdit();
                    break;
                case "vertical":
                    float ydiff = pts.get(0).y-pts.get(pts.size()-1).y;
                    if (ydiff > 50){
                        mListener.moveCursor(3, false);
                    } else if (ydiff < -50){
                        mListener.moveCursor(4, false);
                    }
                    break;
                case "horizontal":
                    float xdiff = pts.get(0).x-pts.get(pts.size()-1).x;
                    if (xdiff > 50){
                        mListener.moveCursor(1, true);
                    } else if (xdiff < -50){
                        mListener.moveCursor(2, true);
                    }
                    break;
                default:
                    return;
            }
        }
    }

    EditGestureRecognizer mGestureRecognizer = new EditGestureRecognizer();

    class EditGestureRecognizer {

        Points ORIGIN = new Points(0, 0, 0);
        int NUMPOINTS = 32;

        ArrayList<PointCloud> mGestures = new ArrayList<PointCloud>();
        ArrayList<Points> mpoints = new ArrayList<Points>();

        class PointCloud {
            public String name;
            public ArrayList<Points> points = null;
            public PointCloud(String name, ArrayList<Points> points){
                this.name = name;
                this.points = resample(points, NUMPOINTS);
                scale(this.points);
                translateTo(this.points);
            }
        }

        public void clear(){
            mpoints.clear();
        }

        public void addPoint(int x, int y){
            mpoints.add(new Points(x, y, 0));
//            Log.e("[Recog]", "addPoint x: "+x+" y: "+y );
        }

        public String recognize(){
            if (mpoints.size() < 4) return "null";
            // user-dependent sample gathering
            if (true) {
                String xs = "";
                String ys = "";
                for (Points p : mpoints) {
                    xs += ((int)p.x+", ");
                    ys += ((int)p.y+", ");
                }
                Log.e("[Points]", "x: "+xs);
                Log.e("[Points]", "y: "+ys);
            }
            mpoints = resample(mpoints, NUMPOINTS);
            scale(mpoints);
            translateTo(mpoints);
            float b = Float.POSITIVE_INFINITY;
            int u = -1;
            for (int i = 0; i < mGestures.size(); ++i){
                float d = greedyCLoudMatch(mpoints, mGestures.get(i));
                if (d <= b){
                    b = d;
                    u = i;
                }
            }

            if (u == -1){
                return "null";
            } else {
                Log.e("[Recog]", "recog: " + mGestures.get(u).name + " score: " + b);
                return mGestures.get(u).name;
            }
        }

        //helper functions
        private float pathLength(ArrayList<Points> points){
            float d = 0;
            for (int i = 1; i  < points.size(); ++i){
                if (points.get(i).time == points.get(i-1).time)
                { d += points.get(i).distanceTo(points.get(i-1)); }
            }
            return d;
        }

        private Points centroid(ArrayList<Points> points){
            float x = 0;
            float y = 0;
            for (int i = 0; i < points.size(); ++i){
                x += points.get(i).x;
                y += points.get(i).y;
            }
            x /= points.size();
            y /= points.size();
            return new Points(x, y, 0);
        }

        private void translateTo(ArrayList<Points> points){
            if (points.size() < 1) return;
            Points c = centroid(points);
            for (int i = 0; i < points.size(); i++){
                float qx = points.get(i).x + ORIGIN.x - c.x;
                float qy = points.get(i).y + ORIGIN.y - c.y;
                long id = points.get(i).time;
                points.set(i, new Points(qx, qy, id));
            }
        }

        private void scale(ArrayList<Points> points){
            if (points.size() < 1) return;
            float minX = Float.POSITIVE_INFINITY;
            float maxX = Float.NEGATIVE_INFINITY;
            float minY = Float.POSITIVE_INFINITY;
            float maxY = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < points.size(); ++i){
                minX = Math.min(minX, points.get(i).x);
                minY = Math.min(minY, points.get(i).y);
                maxX = Math.max(maxX, points.get(i).x);
                maxY = Math.max(maxY, points.get(i).y);
            }
            int size = (int)Math.max(maxX - minX, maxY - minY);
            for (int i = 0; i < points.size(); ++i){
                float qx = (points.get(i).x - minX)/size;
                float qy = (points.get(i).y - minY)/size;
                long id = points.get(i).time;
                points.set(i, new Points(qx, qy, id));
            }
        }

        private ArrayList<Points> resample(ArrayList<Points> points, int n){
            if (points.size() < 1) return points;
            float I = pathLength(points) / (n-1);
            float D = 0;
            ArrayList newpoints = new ArrayList<Points>();
            newpoints.add(points.get(0));
            for (int i = 1; i < points.size(); ++i){
                if (points.get(i).time == points.get(i-1).time){
                    float d = points.get(i-1).distanceTo(points.get(i));
                    if ((D+d)>=I){
                        float qx = points.get(i-1).x + ((I - D) / d) * (points.get(i).x - points.get(i-1).x);
                        float qy = points.get(i-1).y + ((I - D) / d) * (points.get(i).y - points.get(i-1).y);
                        long id = points.get(i).time;
                        Points q = new Points(qx, qy, id);
                        newpoints.add(q);
                        points.add(i, q);
                        D = 0;
                    } else { D += d; }
                }
            }
            if (newpoints.size() == n-1)
            { newpoints.add(points.get(points.size()-1)); }
            return newpoints;
        }

        private float cloudDistance(ArrayList<Points> pts1, ArrayList<Points> pts2, int start){
            boolean[] matched = new boolean[pts1.size()];
            for (int i = 0; i < pts1.size(); ++i){
                matched[i] = false;
            }
            float sum = 0;
            int i = start;
            do {
                int index = -1;
                float min = Float.POSITIVE_INFINITY;
                for (int j = 0; j < matched.length; ++j){
                    if (!matched[j]){
                        float d = pts1.get(i).distanceTo(pts2.get(j));
                        if (d < min){
                            min = d;
                            index = j;
                        }
                    }
                }
                matched[index] = true;
                float weight = 1-((i - start + pts1.size()) % pts1.size()) / pts1.size();
                sum += weight * min;
                i = (i+1) % pts1.size();
            } while (i != start);
            return sum;
        }

        private float greedyCLoudMatch(ArrayList<Points> points, PointCloud cloud){
            float step = (float)Math.floor(Math.pow(points.size(), 0.5));
            float min = Float.POSITIVE_INFINITY;
            for (int i = 0; i < points.size(); i += step){
                float d1 = cloudDistance(points, cloud.points, i);
                float d2 = cloudDistance(cloud.points, points, i);
                min = Math.min(min, Math.min(d1, d2));
            }
            return min;
        }

        public EditGestureRecognizer(){
            //copy gestures
            ArrayList<Points> points = new ArrayList<Points>();
            int[] x = {696, 680, 675, 666, 654, 642, 636, 631, 628, 622, 613, 607, 604, 603, 603, 604, 604, 607, 612, 612, 620, 628};
            int[] y = {208, 215, 217, 226, 237, 250, 256, 265, 268, 279, 300, 319, 333, 334, 347, 365, 367, 380, 389, 389, 401, 409};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("copy", points));
            //2
            points = new ArrayList<Points>();
            x = new int[]{696, 681, 680, 668, 655, 655, 644, 635, 633, 623, 613, 612, 603, 597, 596, 590, 586, 586, 584, 584, 583, 583, 582, 585, 586, 587, 591, 594, 597, 600, 609, 620, 627, 653, 666, 683, 718, 735, 747, 778, 791, 803, 833};
            y = new int[]{177, 183, 183, 193, 205, 206, 215, 223, 226, 238, 250, 252, 265, 277, 280, 295, 306, 310, 317, 324, 335, 338, 345, 354, 359, 362, 371, 375, 378, 381, 386, 389, 392, 399, 402, 398, 397, 395, 395, 388, 387, 387, 376};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("copy", points));
            //3
            points = new ArrayList<Points>();
            x = new int[]{652, 634, 623, 617, 603, 596, 591, 581, 575, 571, 563, 558, 554, 549, 547, 545, 543, 541, 541, 541, 544, 546, 551};
            y = new int[]{131, 141, 147, 152, 160, 165, 170, 179, 184, 189, 199, 206, 214, 221, 234, 244, 256, 267, 276, 296, 318, 327, 340};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("copy", points));
            //4
            points = new ArrayList<Points>();
            x = new int[]{725, 705, 685, 666, 665, 646, 628, 628, 612, 599, 599, 587, 575, 575, 566, 559, 559, 555, 558, 559, 560, 564, 569, 571, 579, 585, 587, 596, 603, 606, 611, 619, 635, 654, 668, 674, 684, 695, 708, 715, 725, 735, 746, 752, 760, 770};
            y = new int[]{122, 123, 126, 133, 133, 144, 154, 155, 166, 179, 180, 197, 220, 222, 248, 274, 276, 290, 304, 319, 322, 338, 351, 355, 369, 378, 381, 391, 397, 399, 403, 406, 411, 412, 409, 409, 407, 403, 396, 393, 388, 379, 369, 364, 356, 346};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("copy", points));
            //5
            points = new ArrayList<Points>();
            x = new int[]{795, 777, 751, 706, 683, 662, 624, 605, 592, 578, 565, 542, 530, 524, 515, 512, 508, 504, 501, 499, 501, 504, 504, 509, 517, 518, 534, 542, 561, 569, 596, 643, 698, 759, 773, 818, 865, 888, 900, 925, 929, 941, 953, 959, 962, 968, 971, 973, 977, 977, 980, 982, 983, 983, 984, 984, 984, 984, 984, 984};
            y = new int[]{214, 220, 234, 260, 273, 288, 316, 330, 343, 354, 371, 401, 416, 427, 440, 449, 455, 468, 487, 496, 502, 514, 516, 523, 530, 531, 538, 542, 544, 545, 547, 545, 539, 531, 529, 521, 513, 509, 507, 502, 501, 497, 492, 489, 488, 484, 482, 481, 479, 478, 477, 475, 474, 474, 473, 472, 472, 471, 471, 470};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("copy", points));
            //user-defined
//            points = new ArrayList<Points>();
//            x = new int[]{};
//            y = new int[]{};
//            for (int i = 0; i < x.length; ++i){
//                points.add(new Points(x[i], y[i], 0));
//            }
//            mGestures.add(new PointCloud("copy", points));

            //paste gestures
            points = new ArrayList<Points>();
            x = new int[]{706, 712, 713, 718, 725, 732, 739, 739, 745, 750, 750, 753, 757, 757, 761, 766, 766, 774, 783, 791, 803};
            y = new int[]{288, 313, 319, 336, 360, 380, 396, 397, 407, 415, 415, 419, 421, 421, 420, 416, 415, 400, 376, 344, 300};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("paste", points));

            //2
            points = new ArrayList<Points>();
            x = new int[]{707, 712, 714, 717, 721, 721, 724, 726, 726, 728, 730, 730, 732, 735, 740, 746, 747, 750, 757, 765, 769, 774, 782, 786, 791, 804, 810, 822, 842, 852, 865};
            y = new int[]{366, 385, 394, 400, 410, 410, 417, 422, 422, 425, 426, 426, 425, 421, 413, 397, 397, 389, 371, 349, 338, 320, 281, 261, 242, 201, 180, 157, 115, 94, 67};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("paste", points));
            //3
            points = new ArrayList<Points>();
            x = new int[]{704, 710, 712, 715, 718, 726, 729, 734, 743, 747, 750, 757, 758, 761, 764, 765, 767, 769, 769, 772, 775, 776, 780, 782, 788};
            y = new int[]{215, 231, 239, 251, 262, 286, 297, 309, 332, 343, 347, 368, 373, 379, 384, 385, 388, 390, 390, 390, 387, 382, 375, 369, 343};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("paste", points));
            //4
            points = new ArrayList<Points>();
            x = new int[]{740, 742, 744, 748, 752, 753, 757, 762, 764, 766, 768, 769, 769, 772, 774, 774, 776, 777, 777, 777, 777, 777, 777, 776, 775, 774, 774, 773, 776, 777, 777, 777, 780, 786, 786, 794, 806, 807};
            y = new int[]{245, 291, 309, 335, 368, 381, 389, 411, 420, 428, 436, 439, 442, 448, 455, 456, 460, 462, 463, 463, 464, 464, 464, 462, 451, 448, 434, 405, 365, 345, 329, 323, 281, 231, 223, 182, 125, 117};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("paste", points));
            //5
            points = new ArrayList<Points>();
            x = new int[]{640, 644, 647, 650, 653, 656, 658, 660, 662, 664, 665, 666, 668, 669, 671, 672, 673, 674, 675, 677, 680, 682, 688, 696, 700, 709, 730, 753, 761, 771, 787, 795, 806, 815, 824, 849, 861, 875, 885, 900, 926, 935, 950, 969, 974};
            y = new int[]{272, 288, 301, 317, 326, 338, 344, 352, 359, 366, 372, 376, 381, 385, 388, 390, 392, 394, 394, 394, 395, 395, 393, 387, 384, 374, 344, 301, 284, 266, 243, 231, 219, 207, 201, 179, 168, 159, 151, 140, 121, 113, 105, 92, 88};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("paste", points));
            //user-defined
            points = new ArrayList<Points>();
            x = new int[]{1310, 1255, 1232, 1203, 1159, 1145, 1125, 1099, 1091, 1078, 1058, 1053, 1044, 1031, 1027, 1019, 1004, 1000, 990, 974, 970, 955, 933, 929, 913, 894, 891, 883, 875, 873, 867};
            y = new int[]{319, 369, 390, 418, 462, 475, 498, 526, 532, 544, 558, 560, 567, 574, 575, 576, 572, 571, 564, 550, 547, 532, 508, 503, 479, 436, 430, 396, 355, 347, 331};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("paste", points));


            //cut gestures
            points = new ArrayList<Points>();
            x = new int[]{786, 768, 757, 751, 742, 736, 723, 716, 712, 706, 702, 693, 688, 685, 676, 671, 668, 664, 660, 653, 649, 647, 641, 638, 635};
            y = new int[]{264, 280, 289, 296, 304, 310, 322, 328, 332, 336, 340, 346, 348, 351, 355, 356, 358, 359, 359, 358, 357, 356, 352, 350, 346};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("cut", points));

            //2
            points = new ArrayList<Points>();
            x = new int[]{789, 776, 773, 762, 747, 744, 732, 719, 716, 707, 696, 694, 686, 678, 677, 671, 664, 663, 656, 648, 647, 641, 633, 632, 626, 620, 619, 613, 605, 604, 603, 603, 603, 605, 608, 612, 616, 617, 622, 626, 627, 629, 631, 635, 637, 640, 645, 646, 649, 653, 661, 665, 670, 678, 680, 691, 702, 706, 713, 722, 737, 741, 750, 762, 787, 793, 812, 827, 834, 860, 873};
            y = new int[]{211, 232, 236, 253, 275, 279, 296, 315, 317, 331, 344, 345, 355, 363, 363, 369, 375, 375, 379, 382, 382, 383, 382, 381, 378, 372, 371, 362, 338, 336, 318, 303, 295, 287, 271, 258, 247, 247, 238, 231, 231, 227, 226, 222, 220, 220, 220, 220, 220, 223, 228, 230, 237, 245, 247, 259, 273, 277, 286, 291, 304, 307, 315, 323, 339, 343, 357, 364, 367, 376, 380};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("cut", points));
            //3
            points = new ArrayList<Points>();
            x = new int[]{796, 788, 781, 771, 761, 751, 743, 725, 716, 710, 696, 684, 673, 663, 654, 646, 639, 632, 629, 623, 613, 610, 604, 595, 592, 587, 580, 578, 572, 564, 562, 559, 556, 555, 557, 560, 560, 564, 570, 570, 576, 583, 583, 590, 593, 600, 600, 613, 628, 628, 647, 667, 667, 686, 704, 704, 722, 740, 741, 760, 781, 782, 804, 825, 827, 848, 864, 866, 882, 896, 898, 910};
            y = new int[]{176, 188, 199, 213, 230, 245, 255, 284, 298, 311, 331, 347, 361, 372, 381, 388, 394, 399, 400, 403, 405, 405, 405, 403, 402, 398, 391, 389, 380, 364, 360, 344, 324, 320, 306, 288, 284, 271, 256, 253, 244, 233, 231, 225, 221, 217, 216, 210, 205, 204, 204, 207, 207, 212, 221, 221, 233, 246, 247, 262, 277, 278, 292, 304, 305, 316, 324, 325, 332, 338, 339, 345};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("cut", points));
            //4
            points = new ArrayList<Points>();
            x = new int[]{811, 779, 753, 749, 715, 689, 687, 662, 643, 639, 620, 608, 605, 597, 592, 582, 580, 574, 568, 560, 558, 553, 550, 546, 544, 539, 536, 536, 534, 533, 532, 531, 531, 532, 535, 536, 541, 553, 559, 572, 598, 611, 631, 667, 689, 703, 735, 751, 763, 789, 800, 809, 826, 833, 837, 850, 859, 869, 881, 887};
            y = new int[]{215, 227, 239, 242, 256, 265, 267, 281, 290, 293, 303, 309, 311, 315, 317, 320, 322, 324, 325, 325, 326, 326, 326, 326, 325, 323, 322, 321, 319, 318, 315, 310, 306, 305, 298, 294, 291, 284, 280, 276, 269, 265, 264, 262, 263, 264, 266, 267, 271, 275, 277, 279, 287, 290, 293, 301, 304, 309, 318, 322};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("cut", points));
            //5
            points = new ArrayList<Points>();
            x = new int[]{852, 838, 821, 802, 795, 782, 767, 751, 742, 729, 718, 706, 696, 685, 676, 658, 649, 640, 621, 611, 604, 596, 584, 567, 555, 549, 547, 539};
            y = new int[]{228, 241, 261, 279, 288, 301, 313, 326, 334, 344, 355, 364, 373, 382, 389, 401, 407, 411, 418, 421, 420, 420, 415, 408, 399, 394, 391, 379};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("cut", points));
            //user-defined
//            points = new ArrayList<Points>();
//            x = new int[]{};
//            y = new int[]{};
//            for (int i = 0; i < x.length; ++i){
//                points.add(new Points(x[i], y[i], 0));
//            }
//            mGestures.add(new PointCloud("cut", points));

            //undo gestures
            points = new ArrayList<Points>();
            x = new int[]{941, 969, 991, 994, 1014, 1025, 1027, 1035, 1040, 1041, 1044, 1046, 1047, 1048, 1046, 1046, 1043, 1039, 1039, 1035, 1031, 1030, 1022, 1012, 1008, 988, 970, 962, 937, 920, 912, 893, 884, 879, 869, 864, 861, 854, 851, 849, 846, 844, 843, 841, 840, 840, 840, 841, 842, 848, 853, 859, 875, 885, 900, 931, 948, 972, 1016, 1035, 1064, 1106, 1107, 1110};
            y = new int[]{287, 292, 298, 299, 305, 308, 309, 311, 311, 312, 313, 313, 314, 316, 318, 319, 322, 324, 325, 328, 330, 331, 336, 340, 343, 354, 364, 369, 387, 398, 405, 419, 426, 431, 440, 446, 450, 458, 462, 466, 472, 474, 477, 481, 483, 485, 489, 491, 493, 496, 496, 497, 497, 497, 498, 499, 499, 500, 502, 502, 504, 505, 505, 505};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("undo", points));

            //2
            points = new ArrayList<Points>();
            x = new int[]{625, 641, 656, 672, 682, 695, 704, 714, 722, 737, 743, 750, 756, 759, 762, 767, 772, 773, 776, 778, 778, 778, 779, 779, 779, 779, 779, 778, 776, 775, 768, 764, 751, 730, 707, 686, 679, 667, 651, 638, 626, 622, 616, 607, 604, 599, 592, 588, 586, 582, 581, 578, 575, 574, 574, 575, 575, 578, 587, 589, 605, 632, 645, 665, 699, 705, 735, 769, 786};
            y = new int[]{122, 120, 119, 118, 118, 117, 116, 115, 115, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 116, 118, 121, 131, 135, 150, 173, 197, 219, 225, 240, 257, 271, 285, 288, 296, 305, 307, 313, 320, 323, 326, 331, 332, 336, 340, 340, 344, 348, 348, 352, 358, 359, 367, 377, 382, 390, 403, 405, 416, 428, 434};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("undo", points));
            //3
            points = new ArrayList<Points>();
            x = new int[]{976, 961, 939, 915, 897, 852, 829, 809, 787, 768, 750, 734, 705, 691, 680, 659, 650, 642, 629, 624, 618, 609, 605, 602, 597, 595, 593, 590, 589, 588, 587, 586, 586, 587, 587, 591, 599, 600, 613, 631, 640, 652, 674, 685, 696, 717, 722, 738, 757, 766, 774, 788, 789, 801, 812, 817, 821, 829, 829, 834, 837, 837, 838, 838, 838, 837, 834, 834, 832, 828, 818, 818, 798, 768, 767, 730, 694, 694, 662, 633, 633, 607, 594, 582, 569, 561, 550, 539, 528};
            y = new int[]{342, 346, 352, 358, 364, 376, 382, 388, 394, 398, 403, 408, 415, 418, 420, 424, 425, 427, 429, 429, 430, 431, 431, 431, 431, 431, 430, 429, 428, 428, 426, 425, 423, 420, 418, 414, 404, 401, 389, 375, 368, 359, 341, 332, 323, 306, 301, 288, 270, 261, 254, 240, 238, 227, 215, 209, 204, 193, 192, 183, 175, 174, 168, 162, 161, 157, 153, 153, 151, 148, 143, 143, 139, 139, 139, 145, 155, 156, 166, 173, 174, 185, 190, 196, 201, 206, 211, 211, 213};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("undo", points));
            //4
            points = new ArrayList<Points>();
            x = new int[]{859, 835, 814, 811, 799, 788, 769, 768, 758, 747, 729, 726, 715, 708, 697, 694, 683, 676, 674, 667, 663, 662, 659, 657, 657, 656, 656, 656, 656, 658, 662, 666, 670, 687, 716, 730, 747, 776, 790, 802, 826, 838, 850, 871, 881, 889, 904, 915, 919, 922, 927, 931, 933, 933, 934, 933, 932, 931, 928, 927, 924, 916, 913, 902, 883, 880, 849, 802, 793, 748, 702, 693, 656, 620, 615, 585, 541, 519, 496};
            y = new int[]{410, 413, 415, 416, 417, 419, 423, 424, 426, 427, 430, 431, 433, 433, 433, 434, 434, 433, 433, 431, 429, 429, 427, 425, 425, 424, 423, 420, 418, 416, 410, 407, 402, 391, 373, 364, 355, 339, 331, 323, 307, 299, 292, 277, 269, 261, 248, 237, 232, 229, 221, 214, 207, 204, 201, 196, 193, 191, 187, 186, 183, 178, 176, 173, 170, 169, 172, 176, 176, 181, 191, 192, 205, 216, 217, 227, 238, 243, 247,};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("undo", points));

            //user-defined
//            points = new ArrayList<Points>();
//            x = new int[]{};
//            y = new int[]{};
//            for (int i = 0; i < x.length; ++i){
//                points.add(new Points(x[i], y[i], 0));
//            }
//            mGestures.add(new PointCloud("undo", points));

            //vertical gestures
            points = new ArrayList<Points>();
            x = new int[]{1018, 1018, 1017, 1017, 1019, 1022, 1023, 1025, 1024, 1024, 1024, 1025, 1026, 1028, 1028, 1029, 1029, 1028, 1028, 1028, 1025, 1024};
            y = new int[]{427, 457, 483, 488, 516, 536, 541, 562, 576, 581, 594, 599, 602, 606, 607, 608, 605, 598, 595, 577, 564, 559};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("vertical", points));

            //2
            points = new ArrayList<Points>();
            x = new int[]{992, 992, 992, 993, 996, 996, 1003, 1011, 1012, 1022, 1031, 1032, 1039, 1048, 1049, 1058, 1069, 1069, 1080, 1090, 1090, 1098, 1097, 1096};
            y = new int[]{439, 403, 392, 363, 322, 312, 277, 238, 230, 200, 174, 169, 153, 136, 133, 112, 89, 86, 67, 55, 54, 49, 49, 49};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("vertical", points));
            //3
            points = new ArrayList<Points>();
            x = new int[]{1020, 1021, 1021, 1021, 1022, 1022, 1023, 1023, 1023, 1025, 1026, 1027, 1028, 1028, 1028, 1027, 1026, 1026, 1025, 1025, 1026, 1022};
            y = new int[]{437, 474, 475, 505, 533, 533, 557, 578, 580, 599, 611, 613, 622, 630, 632, 643, 651, 653, 658, 655, 655, 646};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("vertical", points));
            //4
            points = new ArrayList<Points>();
            x = new int[]{1000, 983, 973, 972, 971, 975, 977, 986, 999, 1002, 1011};
            y = new int[]{352, 280, 223, 212, 159, 130, 124, 97, 72, 68, 55};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("vertical", points));

            //user-defined
//            points = new ArrayList<Points>();
//            x = new int[]{};
//            y = new int[]{};
//            for (int i = 0; i < x.length; ++i){
//                points.add(new Points(x[i], y[i], 0));
//            }
//            mGestures.add(new PointCloud("vertical", points));

            //horizontal gestures
            points = new ArrayList<Points>();
            x = new int[]{975, 944, 912, 912, 880, 849, 849};
            y = new int[]{414, 422, 429, 430, 438, 443, 444};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("horizontal", points));

            //2
            points = new ArrayList<Points>();
            x = new int[]{960, 1034, 1078, 1102, 1157, 1179, 1193, 1214, 1221, 1226, 1234, 1239, 1243};
            y = new int[]{338, 320, 308, 302, 288, 282, 279, 275, 273, 272, 271, 271, 271};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("horizontal", points));
            //3
            points = new ArrayList<Points>();
            x = new int[]{1064, 1129, 1169, 1175, 1204, 1221, 1224, 1244};
            y = new int[]{392, 389, 387, 387, 386, 381, 381, 377};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("horizontal", points));
            //4
            points = new ArrayList<Points>();
            x = new int[]{988, 966, 940, 938, 897, 845, 838, 775, 716, 706, 649};
            y = new int[]{270, 270, 270, 270, 272, 275, 276, 283, 292, 294, 304};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("horizontal", points));

            //user-defined
//            points = new ArrayList<Points>();
//            x = new int[]{};
//            y = new int[]{};
//            for (int i = 0; i < x.length; ++i){
//                points.add(new Points(x[i], y[i], 0));
//            }
//            mGestures.add(new PointCloud("horizontal", points));
        }
    }



}
