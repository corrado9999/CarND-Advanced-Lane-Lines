import numpy as np
import cv2
import sklearn

def polyfit(x, y, deg, w=1, robust=False):
    x, y, w = np.broadcast_arrays(x, y, w)

    if not robust or len(x)-1 < deg+1:
        return np.pad(np.polyfit(x, y, min(deg, len(x)-1), w=w), (max(deg-len(x)+1, 0), 0), mode='constant')

    min_residuals = np.inf
    for i in range(len(x)):
        xx, yy, ww = [np.concatenate([t[:i], t[i+1:]]) for t in (x, y, w)]
        c = np.pad(np.polyfit(xx, yy, min(deg, len(xx)-1), w=ww), (max(deg-len(xx)+1, 0), 0), mode='constant')
        res = np.square(np.polyval(c, xx) - yy).mean() ** 0.5
        if res < min_residuals:
            min_residuals = res
            coeffs = c
    return coeffs

def polyline(base, poly, isClosed=False, filled=False, alpha=1, **kwargs):
    base = np.asarray(base)
    if base.dtype == np.dtype('bool'):
        base = base * 255
    base = np.asarray(base, dtype='uint8')
    canvas = base.copy()
    polys = np.array(poly, dtype='int32').reshape((-1, 1, 2))
    if filled:
        cv2.fillPoly(canvas, [polys], **kwargs)
    else:
        cv2.polylines(canvas, [polys], isClosed, **kwargs)

    return cv2.addWeighted(base, 1-alpha, canvas, alpha, 0)

def fillrects(base, rects, color, alpha=1):
    base = np.asarray(base)
    if base.dtype == np.dtype('bool'):
        base = base * np.uint8(255)
    base = np.asarray(base, dtype='uint8')
    canvas = base.copy()
    for (x,y), w, h in rects:
        cv2.rectangle(canvas, (int(x), int(y)), (int(x+w), int(y+h)), color, cv2.FILLED)

    return cv2.addWeighted(base, 1-alpha, canvas, alpha, 0)


def fitpoly(x, y, deg, w=None, robust=False):
    coeffs = np.polyfit(x, y, deg, w=w)
    if not robust:
        return coeffs
    residuals = np.polyval(coeffs, x) - y
    best = np.argsort(np.abs(residuals))[:deg+1]
    return np.polyfit(x[best], y[best], deg, w=None if w is None else w[best])
