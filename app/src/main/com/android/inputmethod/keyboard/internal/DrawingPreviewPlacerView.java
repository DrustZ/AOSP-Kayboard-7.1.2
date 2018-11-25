/*
 * Copyright (C) 2012 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.inputmethod.keyboard.internal;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.PopupWindow;
import android.widget.RelativeLayout;
import android.widget.TextView;

import com.android.inputmethod.latin.common.CoordinateUtils;

import org.w3c.dom.Text;

import java.util.ArrayList;

import static android.view.Gravity.NO_GRAVITY;

public final class DrawingPreviewPlacerView extends RelativeLayout {
    private final int[] mKeyboardViewOrigin = CoordinateUtils.newInstance();

    private final ArrayList<AbstractDrawingPreview> mPreviews = new ArrayList<>();

    private TextView mIndicatorView = null;

    public DrawingPreviewPlacerView(final Context context, final AttributeSet attrs) {
        super(context, attrs);
        setWillNotDraw(false);

        mIndicatorView = new TextView(context);
        mIndicatorView.setTextAlignment(View.TEXT_ALIGNMENT_CENTER);
        mIndicatorView.setTypeface(Typeface.SERIF, Typeface.BOLD);
        mIndicatorView.setTextSize(19);
        mIndicatorView.setTextColor(0xbf339966);
        mIndicatorView.setGravity(Gravity.CENTER);
    }

    public void setHardwareAcceleratedDrawingEnabled(final boolean enabled) {
        if (!enabled) return;
        final Paint layerPaint = new Paint();
        layerPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.SRC_OVER));
        setLayerType(LAYER_TYPE_HARDWARE, layerPaint);
    }

    public void addPreview(final AbstractDrawingPreview preview) {
        if (mPreviews.indexOf(preview) < 0) {
            mPreviews.add(preview);
        }
    }

    public void showIndicatorViewWithText(String text, int x, int y) {
        mIndicatorView.setText(text);
        Rect bounds = new Rect();
        Paint textPaint = mIndicatorView.getPaint();
        textPaint.getTextBounds(text, 0, text.length(), bounds);
        int height = bounds.height();
        int width = bounds.width();
        mIndicatorView.setX(x-width/2-10);
        mIndicatorView.setY(y-height-150);
        mIndicatorView.setHeight(height+80);
        mIndicatorView.setWidth(width+20);
        mIndicatorView.setBackgroundColor(Color.argb(100, 200, 200, 200));
        this.addView(mIndicatorView);
    }

    public void setKeyboardViewGeometry(final int[] originCoords, final int width,
            final int height) {
        CoordinateUtils.copy(mKeyboardViewOrigin, originCoords);
        final int count = mPreviews.size();
        for (int i = 0; i < count; i++) {
            mPreviews.get(i).setKeyboardViewGeometry(originCoords, width, height);
        }
    }

    public void deallocateMemory() {
        final int count = mPreviews.size();
        for (int i = 0; i < count; i++) {
            mPreviews.get(i).onDeallocateMemory();
        }
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        deallocateMemory();
    }

    @Override
    public void onDraw(final Canvas canvas) {
        super.onDraw(canvas);
        final int originX = CoordinateUtils.x(mKeyboardViewOrigin);
        final int originY = CoordinateUtils.y(mKeyboardViewOrigin);
        canvas.translate(originX, originY);
        final int count = mPreviews.size();
        for (int i = 0; i < count; i++) {
            mPreviews.get(i).drawPreview(canvas);
        }
        canvas.translate(-originX, -originY);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (mIndicatorView.getParent() == null) {
            return false;
        }

        final int action = event.getAction();
        final long eventTime = event.getEventTime();
        final int rawX = (int)event.getRawX();
        final int rawY = (int)event.getRawY();
        switch (action) {
            case MotionEvent.ACTION_DOWN:
            case MotionEvent.ACTION_POINTER_DOWN:
//                removeIndicatorView();
                break;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_POINTER_UP:
                removeIndicatorView();
                break;
            case MotionEvent.ACTION_MOVE:
                if (rawY > CoordinateUtils.y(mKeyboardViewOrigin) + 30){
                    removeIndicatorView();
                } else {
                    mIndicatorView.setX(rawX-mIndicatorView.getWidth()/2-10);
                    mIndicatorView.setY(rawY-mIndicatorView.getHeight()-150);
                    mIndicatorView.requestLayout();
                }
                break;
        }
        return true;
    }

    public void removeIndicatorView(){
        this.removeView(mIndicatorView);
    }
}
