package com.android.inputmethod.keyboard;

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
    long lastAngle_time = 0;
    int change_count = 0; // count for reverse direction change
    int initial_move_count = 0; // count for initial cursor moves

    float accumAngle = 0; //total angle moved
    float velocity = 0;
    int pointsToDetectLine = 5;
    int maxPoints = 10;

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

        centerx = ((x1*x1+y1*y1)*(y2-y3) + (x2*x2+y2*y2)*(y3-y1) + (x3*x2+y3*y2)*(y1-y2))/d;
        centery = ((x1*x1+y1*y1)*(x3-x2) + (x2*x2+y2*y2)*(x1-x3) + (x3*x2+y3*y2)*(x2-x1))/d;
        float r = (float)Math.sqrt((centerx-x1)*(centerx-x1)+(centery-y1)*(centery-y1));

        centerx = (float)(centerx*0.5 + x3*0.5);
        centery = (float)(centery*0.5 + y3*0.5);
        if (r > 600 || centerx < 50 || centery < 50 || centerx > 950 || centery > 550){ //too large
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
        boolean hasCenter = getCircleCenter(pts.get(0), pts.get(maxPoints/2), lastpoint);
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

            if (lastcenter == null && diff > 20){
                lastcenter = newcenter;
                backupCenter = null;
                KBView.updateCenter((int)lastcenter.x, (int)lastcenter.y, 0xFF00FF00);
            }

            if (lastcenter == null && backupCenter == null) {
                vfilter.clear();
                return;
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
            lastAngle_time = lastpoint.time;
            lastAngle = a;
        }
        float diff = diffTwoAngles(a, lastAngle);

        int threshold = 40;
        if (linedirection <= 2){
            threshold = 30;
//            if (initial_move_count < 5){
//                threshold = 35;
//            }
        }
        if (Math.abs(diff) > threshold){
            float velocity = 100*Math.abs(diff)/(lastpoint.time-lastAngle_time);
            velocity = vfilter.process(velocity);
            initial_move_count += 1;
            if (initial_move_count < 5) {
                velocity = 0;
            }
            if ((diff < 0 && curDirection == 1) || (diff > 0 && curDirection == -1)){
                if (Math.abs(diff) < 20+threshold){
                change_count += 1;
//                Log.e("[Diff]", "move diff"+diff+ " angle now "+a);
                if (change_count > 1){
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
            lastAngle_time = lastpoint.time;
        }

        if (ellipseLocated <= 1 && Math.abs(accumAngle) > 180 && cPoints.size() > 5){
            if (locateEllipse())
                ellipseLocated = 2;
        }

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
        if (elipse.center.x < 50 || elipse.center.y < 50 || elipse.center.x > 950 || elipse.center.y > 550)
            return false;
        lastcenter = elipse.center;
        KBView.updateCenter((int)lastcenter.x, (int)lastcenter.y, 0xFF00FFFF);
        return true;
    }

    //the main logic: get points and process
    public void processGestureEvent(int x, int y, long eventime){
        if (notRingmode) return;
        int size = pts.size();
        if (size == 0) {
            pts.add(new Points(x, y, eventime));
            gestureStartTime = eventime;
            return;
        }
        //too short
        if (eventime-pts.get(size-1).time < 30) return;
        pts.add(new Points(x, y, eventime));
        if (pts.size() > maxPoints) pts.remove(0);

        //for gesture recognizer
        if (eventime - gestureStartTime < 500){
            mGestureRecognizer.addPoint(x, y);
        }

        if (!ringmodeEntered && size >= pointsToDetectLine){
            boolean lineres = false;
            // we ignore slight movement
            if (pts.get(pts.size()-1).distanceTo(pts.get(pts.size()-5)) < 15) {
                return;
            }
            if (size == pointsToDetectLine){
                lineres = detectLine(pointsToDetectLine);
            } else {
                lineres = detectLine(pointsToDetectLine*2);
            }
            if (!lineres && !lineDetected){
                notRingmode = true;//the gesture doesn't begin with a line, so we abandon it
                if (mListener != null) mListener.enteringRingMode(false);
                Log.e("[Log]", "not entering ring mode" );
            } else if (lineres){
                lineDetected = true;
            } else {
                for (int i = 0; i < pts.size()-8; ++i){
                    pts.remove(0);
                }
                ringmodeEntered = true;//no longer line, entering ring mode
                lastcenter = new org.opencv.core.Point(pts.get(0).x, pts.get(0).y);
                KBView.updateCenter((int)lastcenter.x, (int)lastcenter.y, 0xFF00FF00);
                mListener.kbVibrate();
                if (mListener != null) mListener.enteringRingMode(true);
//                Log.e("[Log]", "entering circle mode");
            }
        }
        //if ringmode then we process the points
        else if (ringmodeEntered && pts.size() >= maxPoints){
            cPoints.add(new org.opencv.core.Point(x, y));
            //other gestures have 500 ms to accomplish
            if (eventime - gestureStartTime > 500) {
                processRingPoints();
            }
        }
    }

    private void moveCursor(float diff, float velocity){
        if (mListener != null){
//            Log.e("[Log]", "Velo " + velocity );
            int movedirection = 0;
            if (!ringmodeEntered){
                mListener.moveCursor(linedirection, false);
            }
            if (linedirection == 1 || linedirection == 2){
                if (curDirection == 1){ // clockwise: right
                    movedirection = 2;
                } else {
                    movedirection = 1;
                }
            } else {
                if (curDirection == 1){ // clockwise: up
                    movedirection = 3;
                } else {
                    movedirection = 4;
                }
            }
            //CD-gain
            int movetime = 1;
            if (curDirection <= 2 && velocity > 35){
                movetime = (int)Math.ceil(5*(1-Math.exp(0.04*(35-velocity))));
            }
            // word-level
//            if (curDirection <= 2 && velocity > 20){
//                mListener.moveCursor(movedirection, true);
//            } else {
//                mListener.moveCursor(movedirection, false);
//            }
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
            mpoints = resample(mpoints, NUMPOINTS);
            scale(mpoints);
            translateTo(mpoints);
            float b = Float.POSITIVE_INFINITY;
            int u = -1;
            for (int i = 0; i < mGestures.size(); ++i){
                float d = greedyCLoudMatch(mpoints, mGestures.get(i));
                if (d < b){
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
            int[] x = {534,466,418,369,353,362,415,526,667};
            int[] y = {95 ,107,151,263,353,403,437,449,425};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("copy", points));
            //2
            points = new ArrayList<Points>();
            x = new int[]{572,532,531,554,602};
            y = new int[]{156,215,277,309,324};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("copy", points));
            //3
            points = new ArrayList<Points>();
            x = new int[]{723,665,619,606,624,648,704};
            y = new int[]{109,134,184,242,301,326,356};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("copy", points));
            //4
            points = new ArrayList<Points>();
            x = new int[]{695,634,609,632,731};
            y = new int[]{195,270,356,430,502};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("copy", points));
            //5
            points = new ArrayList<Points>();
            x = new int[]{746,707,701,758};
            y = new int[]{195,243,291,327};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("copy", points));

            //paste gestures
            points = new ArrayList<Points>();
            x = new int[]{678,713,759,811};
            y = new int[]{357,407,401,319};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("paste", points));

            //2
            points = new ArrayList<Points>();
            x = new int[]{788,819,839,868,894};
            y = new int[]{318,350,357,277,197};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("paste", points));
            //3
            points = new ArrayList<Points>();
            x = new int[]{783,797,801,810,845};
            y = new int[]{330,402,423,397,196};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("paste", points));
            //4
            points = new ArrayList<Points>();
            x = new int[]{642,689,711,723,745,789,852,885,903,911,913,914};
            y = new int[]{306,387,417,427,420,362,236,166,132,114,107,103};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("paste", points));
            //5
            points = new ArrayList<Points>();
            x = new int[]{822,872,929,986};
            y = new int[]{430,427,276,110};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("paste", points));

            //cut gestures
            points = new ArrayList<Points>();
            x = new int[]{776,694,627,585,568};
            y = new int[]{293,346,361,350,332};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("cut", points));

            //2
            points = new ArrayList<Points>();
            x = new int[]{760,687,647,603,576,568,583,611,677,772,835,882};
            y = new int[]{274,354,388,393,366,290,233,205,211,323,400,433};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("cut", points));
            //3
            points = new ArrayList<Points>();
            x = new int[]{767,684,649,634,634,673,740,790,829,862};
            y = new int[]{276,332,349,345,317,275,252,272,315,346};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("cut", points));
            //4
            points = new ArrayList<Points>();
            x = new int[]{720,644,586,547,538};
            y = new int[]{310,365,378,356,304};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("cut", points));
            //5
            points = new ArrayList<Points>();
            x = new int[]{751,660,583,551,569};
            y = new int[]{297,385,402,378,294};
            for (int i = 0; i < x.length; ++i){
                points.add(new Points(x[i], y[i], 0));
            }
            mGestures.add(new PointCloud("cut", points));
        }
    }



}
