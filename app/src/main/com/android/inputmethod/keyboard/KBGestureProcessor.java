package com.android.inputmethod.keyboard;

import android.graphics.Point;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;

public class KBGestureProcessor {
    final float Epsilon = (float) 1e-3;
    final float Infinite = (float) 1e10;
    private KeyboardActionListener mListener = null;

    ArrayList<Points> pts = new ArrayList<Points>();
    //line detection
    float beta = 0; //slope
    float alpha = 0; //intersection
    float rsquare = 0;

    //logic flags
    boolean lineDetected = false;
    boolean ringmodeEntered = false;
    boolean notRingmode = false; // check if the user doesn't perform a ring gesture

    //control vars
    int linedirection = 0; // no 0 left 1 right 2 up 3 down 4
    int curDirection = 0; // -1 counter clock wise 1 clock wise
    float lastAngle = Infinite; //last moved angle, should be smaller than 360
    long lastAngle_time = 0;
    int change_count = 0;

    float accumAngle = 0; //total angle moved
    float velocity = 0;
    int pointsToDetectLine = 5;
    int maxPoints = 15;

    Points lastcenter = null; //last circle center
    Points backupCenter = null; // in case always can't find a good center in the ringmode
    float centerx = Infinite;
    float centery = Infinite;

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
        lineDetected = false;
        ringmodeEntered = false;
        notRingmode = false;

        linedirection = 0;
        curDirection = 0;
        lastcenter = null;
        centerx = Infinite;
        centery = Infinite;

        lastAngle = Infinite;
        change_count = 0;
        accumAngle = 0;
        velocity = 0;
        pts.clear();

        xfilter.clear();
        yfilter.clear();
        vfilter.clear();
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
        Log.e("[Log]", "detectLine: r2 "+rs1 + " slope "+beta);
        linearRegress(ys, xs); //do twice to avoid failure on vertical condition
        float rs2 = rsquare;
        Log.e("[Log]", "detectLine: r2 "+rs2 + " slope "+beta);

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
            Log.e("[Log]", "line detected"+linedirection);
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

//        Log.e("[Diff]", "getCircleCenter: x "+ centerx + " y " + centery + " radius "+r );
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
                centerx = lastcenter.x;
                centery = lastcenter.y;
            }
        } else {
            centerx = xfilter.process(centerx);
            centery = yfilter.process(centery);
            Points newcenter = new Points(centerx, centery);
            //if the movement is large or not
            float diff = Math.abs(diffTwoAngles(getAngle(lastpoint.x, lastpoint.y, centerx, centery),
                    getAngle(pts.get(maxPoints - 5).x, pts.get(maxPoints - 5).y, centerx, centery)));
            if (diff > 20){
                backupCenter = newcenter;
            }

            if (lastcenter != null && diff > 20){
                if (Math.abs(centerx-lastcenter.x)>300 || Math.abs(centery-lastcenter.y) > 200){
                    lastcenter = newcenter;
                    backupCenter = null;
                    Log.e("[Log]", "aa new center: x "+centerx + " y "+ centery);
                }
            }

            if (lastcenter == null && diff > 20){
                lastcenter = newcenter;
                backupCenter = null;
                Log.e("[Log]", "new center: x "+centerx + " y "+ centery);
            }

            if (lastcenter == null && backupCenter == null) {
                vfilter.clear();
                return;
            }
            else if (lastcenter == null && backupCenter != null){
                lastcenter = backupCenter;
                backupCenter = null;
                Log.e("[Log]", "new center: x "+centerx + " y "+ centery);
            }
        }

        float a = getAngle(lastpoint.x, lastpoint.y, lastcenter.x, lastcenter.y);
        if (lastAngle == Infinite){
            lastAngle = getAngle(pts.get(0).x, pts.get(0).y, lastcenter.x, lastcenter.y);
            lastAngle_time = pts.get(0).time;
        }
        float diff = diffTwoAngles(a, lastAngle);

//        Log.e("[Angle]", ""+a);

        if (Math.abs(diff) > 5){
            float velocity = 100*Math.abs(diff)/(lastpoint.time-lastAngle_time);
            velocity = vfilter.process(velocity);
            if ((diff > -20 && diff < -1 && curDirection == 1) || (diff < 20 && diff > 1 && curDirection == -1)){
                change_count += 1;
//                Log.e("[Diff]", "move diff"+diff+ " angle now "+a);
                if (change_count > 5){
                    //lets change direction
                    change_count = 0;
                    curDirection = -curDirection;
                    //move opposite
                    Log.e("[Log]", "Direction change!" );
                    moveCursor(diff, velocity);
                } else return;
            } else {
                change_count = 0;
                //moving in current direction
                moveCursor(diff, velocity);
                if (diff > 0) curDirection = 1;
                else curDirection = -1;
            }
            accumAngle += diff;
            lastAngle = a;
            lastAngle_time = lastpoint.time;
        }

        if (accumAngle >= 360 || accumAngle <= -360){
            lastcenter = null;
            Log.e("[Log]", "center nulled!");
            accumAngle = 0;
        }
    }

    //the main logic: get points and process
    public void processGestureEvent(int x, int y, long eventime){
        if (notRingmode) return;
        int size = pts.size();
        if (size == 0) {
            pts.add(new Points(x, y, eventime));
            return;
        }
        //too short
        if (eventime-pts.get(size-1).time < 30) return;
        pts.add(new Points(x, y, eventime));
        if (pts.size() > maxPoints) pts.remove(0);

        if (!ringmodeEntered && size >= pointsToDetectLine){
            boolean lineres = false;
            if (size == pointsToDetectLine){
                lineres = detectLine(pointsToDetectLine);
            } else {
                lineres = detectLine(pointsToDetectLine*2);
            }
            if (!lineres && !lineDetected){
                notRingmode = true;//the gesture doesn't begin with a line, so we abandon it
                Log.e("[Log]", "not entering ring mode" );
            } else if (lineres){
                lineDetected = true;
            } else {
                ringmodeEntered = true;//no longer line, entering ring mode
                Log.e("[Log]", "entering circle mode");
            }
        }
        //if ringmode then we process the points
        else if (ringmodeEntered && pts.size() >= maxPoints){
            processRingPoints();
        }
    }

    private void moveCursor(float diff, float velocity){
        if (mListener != null){
            int movedirection = 0;
            if (!ringmodeEntered){
                mListener.moveCursor(linedirection);
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
            mListener.moveCursor(movedirection);
            String dir = "left";
            if (movedirection == 2) dir = "right";
            if (movedirection == 3) dir = "up";
            if (movedirection == 4) dir = "down";
//            Log.e("[Log]", "Moving direction: "+dir);
        }
    }

}
