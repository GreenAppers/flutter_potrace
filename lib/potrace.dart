/* Copyright (C) 2001-2013 Peter Selinger.
 *
 * Flutter port of Potrace (http://potrace.sourceforge.net).
 * 
 * Licensed under the GPL
 */

import 'dart:math' as Math;
import 'dart:typed_data';
import 'dart:ui';

import 'package:image/image.dart' as img;

/**
 * Create SVG from image [src]
 */

String potrace(img.Image src, {
  String turnpolicy = "minority",
  int turdsize = 2,
  bool optcurve = true,
  num alphamax = 1,
  num opttolerance = 0.2,
}) {
  _Bitmap bm = _Bitmap.fromImg(src);
  List<_Path> pathlist = _BitmapToPathlist(bm, turnpolicy, turdsize).run();
  _ProcessPath(optcurve, alphamax, opttolerance).run(pathlist);
  return _GetSVG(bm, 1.0, "", pathlist).run();
}

/**
 * Create Flutter Path from mask [src]
 */

Path potraceMask(Uint8List src, int width, int height, {
  String turnpolicy = "minority",
  int turdsize = 2,
  bool optcurve = true,
  num alphamax = 1,
  num opttolerance = 0.2,
}) {
  _Bitmap bm = _Bitmap.fromData(width, height, src);
  List<_Path> pathlist = _BitmapToPathlist(bm, turnpolicy, turdsize).run();
  _ProcessPath(optcurve, alphamax, opttolerance).run(pathlist);
  return _GetPath(bm, 1.0, "", pathlist).run();
}

/**
 * Peter Selinger (2003). Potrace: a polygon-based tracing algorithm.
 * http://potrace.sourceforge.net/potrace.pdf
 */

class _Point {
  int x, y;
  _Point([this.x=0, this.y=0]);
  _Point copy() => _Point(x, y);
}

class _DPoint {
  num x, y;
  _DPoint([this.x=0, this.y=0]);
  _DPoint.fromPoint(_Point p) : this(p.x, p.y);
  _DPoint copy() => _DPoint(x, y);
}

class _Bitmap {
  int w, h, size;
  Uint8List data;

  _Bitmap(this.w, this.h) : size=w*h, data=Uint8List(w*h);

  _Bitmap.fromData(this.w, this.h, this.data) : size=w*h;

  _Bitmap.fromImg(img.Image image) : w=image.width, h=image.height, size=image.width*image.height, data=Uint8List(image.width*image.height) {
    for (int i = 0; i < image.length; i++) {
      int pixel = image[i];
      num color = 0.2126 * img.getRed(pixel) + 0.7153 * img.getGreen(pixel) + 0.0721 * img.getBlue(pixel);
      data[i] = (color < 128 ? 1 : 0);
    }
  }

  bool at(int x, int y) => (x >= 0 && x < w && y >=0 && y < h) && data[w * y + x] == 1;

  _Point index(num i) {
    _Point point = _Point();
    point.y = (i / w).floor();
    point.x = i - point.y * w;
    return point;
  }

  void flip(int x, int y) {
    if (at(x, y)) {
      data[w * y + x] = 0;
    } else {
      data[w * y + x] = 1;
    }
  }

  _Bitmap copy() {
    _Bitmap bm = _Bitmap(w, h);
    for (int i = 0; i < size; i++) {
      bm.data[i] = data[i];
    }
    return bm;
  }
}

class _Curve {
  int n;
  List<String> tag;
  List<_DPoint> vertex, c;
  bool alphacurve = false;
  List<num> alpha, alpha0, beta;

  _Curve(this.n) : tag=List(n), c=List(n*3), vertex=List(n), alpha=List(n), alpha0=List(n), beta=List(n);
}

class _Path {
  int len = 0;
  List<_Point> pt = <_Point>[];
  List<int> lon = <int>[];
  int x0, y0, m;
  List<_Sum> sums = <_Sum>[];
  List<int> po = <int>[];
  _Curve curve;

  num area = 0, minX = 100000, minY = 100000, maxX= -1, maxY = -1;
  String sign;
}

class _Quad {
  List<num> data = List<num>(9);
  num at(int x, int y) => data[x * 3 + y];
}

class _Opti {
  num pen = 0, t = 0, s = 0, alpha = 0;
  List<_DPoint> c = List<_DPoint>(2);
}

class _Sum {
  num x, y, xy, x2, y2;
  _Sum([this.x=0, this.y=0, this.xy=0, this.x2=0, this.y2=0]);
}

class _BitmapToPathlist {
  _Bitmap bm, bm1;
  String turnpolicy;
  int turdsize;
  _BitmapToPathlist(_Bitmap bM, this.turnpolicy, this.turdsize) : bm=bM, bm1=bM.copy();

  _Point findNext(_Point point) {
    int i = bm1.w * point.y + point.x;
    while (i < bm1.size && bm1.data[i] != 1) {
      i++;
    }
    return i < bm1.size ? bm1.index(i) : null;
  }

  bool majority(int x, int y) {
    for (int i = 2; i < 5; i++) {
      int ct = 0;
      for (int a = -i + 1; a <= i - 1; a++) {
        ct += bm1.at(x + a, y + i - 1) ? 1 : -1;
        ct += bm1.at(x + i - 1, y + a - 1) ? 1 : -1;
        ct += bm1.at(x + a - 1, y - i) ? 1 : -1;
        ct += bm1.at(x - i, y + a) ? 1 : -1;
      }
      if (ct > 0) {
        return true;
      } else if (ct < 0) {
        return false;
      }
    }
    return false;
  }

  _Path findPath(_Point point) {
    _Path path = _Path();
    int x = point.x, y = point.y, dirx = 0, diry = 1, tmp;

    path.sign = bm.at(point.x, point.y) ? "+" : "-";

    while (true) {
      path.pt.add(_Point(x, y));
      if (x > path.maxX)
        path.maxX = x;
      if (x < path.minX)
        path.minX = x;
      if (y > path.maxY)
        path.maxY = y;
      if (y < path.minY)
        path.minY = y;
      path.len++;

      x += dirx;
      y += diry;
      path.area -= x * diry;

      if (x == point.x && y == point.y)
        break;

      bool l = bm1.at(x + (dirx + diry - 1) ~/ 2, y + (diry - dirx - 1) ~/ 2);
      bool r = bm1.at(x + (dirx - diry - 1) ~/ 2, y + (diry + dirx - 1) ~/ 2);

      if (r && !l) {
        if (turnpolicy == "right" ||
        (turnpolicy == "black" && path.sign == '+') ||
        (turnpolicy == "white" && path.sign == '-') ||
        (turnpolicy == "majority" && majority(x, y)) ||
        (turnpolicy == "minority" && !majority(x, y))) {
          tmp = dirx;
          dirx = -diry;
          diry = tmp;
        } else {
          tmp = dirx;
          dirx = diry;
          diry = -tmp;
        }
      } else if (r) {
        tmp = dirx;
        dirx = -diry;
        diry = tmp;
      } else if (!l) {
        tmp = dirx;
        dirx = diry;
        diry = -tmp;
      }
    }
    return path;
  }

  void xorPath(_Path path) {
    int y1 = path.pt[0].y, len = path.len;

    for (int i = 1; i < len; i++) {
      int x = path.pt[i].x;
      int y = path.pt[i].y;

      if (y != y1) {
        int minY = y1 < y ? y1 : y;
        int maxX = path.maxX;
        for (int j = x; j < maxX; j++) {
          bm1.flip(j, minY);
        }
        y1 = y;
      }
    }
  }
 
  List<_Path> run() {
    List<_Path> pathlist = <_Path>[];

    _Point currentPoint = _Point();
    while ((currentPoint = findNext(currentPoint)) != null) {

      _Path path = findPath(currentPoint);
      xorPath(path);

      if (path.area > turdsize)
        pathlist.add(path);
    }

    return pathlist;
  }
}

class _ProcessPath {
  bool optcurve;
  num alphamax, opttolerance;

  _ProcessPath(this.optcurve, this.alphamax, this.opttolerance);

  static int mod(int a, int n) {
    return a >= n ? a % n : a>=0 ? a : n-1-(-1-a) % n;
  }

  static int xprod(_Point p1, _Point p2) {
    return p1.x * p2.y - p1.y * p2.x;
  }

  static bool cyclic(int a, int b, int c) {
    if (a <= c) {
      return (a <= b && b < c);
    } else {
      return (a <= b || b < c);
    }
  }

  static num sign(num i) {
    return i > 0 ? 1 : i < 0 ? -1 : 0;
  }

  static num quadform(_Quad Q, _DPoint w) {
    List<num> v = <num>[ w.x, w.y, 1 ];
    num sum = 0.0;
  
    for (int i=0; i<3; i++) {
      for (int j=0; j<3; j++) {
        sum += v[i] * Q.at(i, j) * v[j];
      }
    }
    return sum;
  }

  static _DPoint interval(num lambda, _DPoint a, _DPoint b) {
    return _DPoint(a.x + lambda * (b.x - a.x), a.y + lambda * (b.y - a.y));
  }

  static _Point dorthInfty(_DPoint p0, _DPoint p2) {
    return _Point(sign(p2.x - p0.x), -sign(p2.y - p0.y));
  }

  static num ddenom(_DPoint p0, _DPoint p2) {
    _Point r = dorthInfty(p0, p2);
    return r.y * (p2.x - p0.x) - r.x * (p2.y - p0.y);
  }

  static num dpara(_DPoint p0, _DPoint p1, _DPoint p2) {
    num x1 = p1.x - p0.x;
    num y1 = p1.y - p0.y;
    num x2 = p2.x - p0.x;
    num y2 = p2.y - p0.y;

    return x1 * y2 - x2 * y1;
  }

  static num cprod(_DPoint p0, _DPoint p1, _DPoint p2, _DPoint p3) {
    num x1 = p1.x - p0.x;
    num y1 = p1.y - p0.y;
    num x2 = p3.x - p2.x;
    num y2 = p3.y - p2.y;

    return x1 * y2 - x2 * y1;
  }

  static num iprod(_DPoint p0, _DPoint p1, _DPoint p2) {
    num x1 = p1.x - p0.x;
    num y1 = p1.y - p0.y;
    num x2 = p2.x - p0.x;
    num y2 = p2.y - p0.y;

    return x1*x2 + y1*y2;
  }

  static num iprod1(_DPoint p0, _DPoint p1, _DPoint p2, _DPoint p3) {
    num x1 = p1.x - p0.x;
    num y1 = p1.y - p0.y;
    num x2 = p3.x - p2.x;
    num y2 = p3.y - p2.y;

    return x1 * x2 + y1 * y2;
  }

  static num ddist(_DPoint p, _DPoint q) {
    return Math.sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y));
  }

  static _DPoint bezier(num t, _DPoint p0, _DPoint p1, _DPoint p2, _DPoint p3) {
    num s = 1 - t;
    return _DPoint(
      s*s*s*p0.x + 3*(s*s*t)*p1.x + 3*(t*t*s)*p2.x + t*t*t*p3.x,
      s*s*s*p0.y + 3*(s*s*t)*p1.y + 3*(t*t*s)*p2.y + t*t*t*p3.y,
    );
  }

  static num tangent(_DPoint p0, _DPoint p1, _DPoint p2, _DPoint p3, _DPoint q0, _DPoint q1) {
    num A = cprod(p0, p1, q0, q1);
    num B = cprod(p1, p2, q0, q1);
    num C = cprod(p2, p3, q0, q1);

    num a = A - 2 * B + C;
    num b = -2 * A + 2 * B;
    num c = A;

    num d = b * b - 4 * a * c;

    if (a==0 || d<0) {
      return -1.0;
    }

    num s = Math.sqrt(d);

    num r1 = (-b + s) / (2 * a);
    num r2 = (-b - s) / (2 * a);

    if (r1 >= 0 && r1 <= 1) {
      return r1;
    } else if (r2 >= 0 && r2 <= 1) {
      return r2;
    } else {
      return -1.0;
    }
  }

  static void calcSums(_Path path) {
    path.x0 = path.pt[0].x;
    path.y0 = path.pt[0].y;

    path.sums = <_Sum>[];
    List<_Sum> s = path.sums;
    s.add(_Sum(0, 0, 0, 0, 0));

    for (int i = 0; i < path.len; i++) {
      int x = path.pt[i].x - path.x0;
      int y = path.pt[i].y - path.y0;
      s.add(_Sum(s[i].x + x, s[i].y + y, s[i].xy + x * y, s[i].x2 + x * x, s[i].y2 + y * y));
    }
  }
 
  static void calcLon(_Path path) {
    int dir, n = path.len;
    List<_Point> pt = path.pt;
    List<num> pivk = List(n), nc = List(n), ct = List(4);
    path.lon = List(n);

    List<_Point> constraint = <_Point>[ _Point(), _Point() ];
    _Point cur = _Point(), off = _Point(), dk = _Point();
    bool foundk;

    int i, j, k1, a, b, c, d, k = 0;
    for( i = n - 1; i >= 0; i--){
      if (pt[i].x != pt[k].x && pt[i].y != pt[k].y) {
        k = i + 1;
      }
      nc[i] = k;
    }
    
    for (i = n - 1; i >= 0; i--) {
      ct[0] = ct[1] = ct[2] = ct[3] = 0;
      dir = ((3 + 3 * (pt[mod(i + 1, n)].x - pt[i].x) + (pt[mod(i + 1, n)].y - pt[i].y)) / 2).floor();
      ct[dir]++;
      
      constraint[0].x = 0;
      constraint[0].y = 0;
      constraint[1].x = 0;
      constraint[1].y = 0;
      
      k = nc[i];
      k1 = i;
      while (true) {
        foundk = false;
        dir = ((3 + 3 * sign(pt[k].x - pt[k1].x) + sign(pt[k].y - pt[k1].y)) / 2).floor();
        ct[dir]++;
        
        if (ct[0] != 0 && ct[1] != 0 && ct[2] != 0 && ct[3] != 0) {
          pivk[i] = k1;
          foundk = true;
          break;
        }
        
        cur.x = pt[k].x - pt[i].x;
        cur.y = pt[k].y - pt[i].y;
        
        if (xprod(constraint[0], cur) < 0 || xprod(constraint[1], cur) > 0) {
          break;
        }
            
        if (cur.x.abs() <= 1 && cur.y.abs() <= 1) {
        
        } else {
          off.x = cur.x + ((cur.y >= 0 && (cur.y > 0 || cur.x < 0)) ? 1 : -1);
          off.y = cur.y + ((cur.x <= 0 && (cur.x < 0 || cur.y < 0)) ? 1 : -1);
          if (xprod(constraint[0], off) >= 0) {
            constraint[0].x = off.x;
            constraint[0].y = off.y;
          }
          off.x = cur.x + ((cur.y <= 0 && (cur.y < 0 || cur.x < 0)) ? 1 : -1);
          off.y = cur.y + ((cur.x >= 0 && (cur.x > 0 || cur.y < 0)) ? 1 : -1);
          if (xprod(constraint[1], off) <= 0) {
            constraint[1].x = off.x;
            constraint[1].y = off.y;
          }
        }
        k1 = k;
        k = nc[k1];
        if (!cyclic(k, i, k1)) {
          break;
        }
      }
      if (foundk == false) {
        dk.x = sign(pt[k].x-pt[k1].x);
        dk.y = sign(pt[k].y-pt[k1].y);
        cur.x = pt[k1].x - pt[i].x;
        cur.y = pt[k1].y - pt[i].y;

        a = xprod(constraint[0], cur);
        b = xprod(constraint[0], dk);
        c = xprod(constraint[1], cur);
        d = xprod(constraint[1], dk);

        j = 10000000;
        if (b < 0) {
          j = (a / -b).floor();
        }
        if (d > 0) {
          j = Math.min(j, (-c / d).floor());
        }
        pivk[i] = mod(k1+j,n);
      }
    }

    j=pivk[n-1];
    path.lon[n-1]=j;
    for (i=n-2; i>=0; i--) {
      if (cyclic(i+1,pivk[i],j)) {
        j=pivk[i];
      }
      path.lon[i]=j;
    }

    for (i=n-1; cyclic(mod(i+1,n),j,path.lon[i]); i--) {
      path.lon[i] = j;
    }
  }

  static num penalty3(_Path path, int i, int j) {
    int n = path.len;
    List<_Point> pt = path.pt;
    List<_Sum> sums = path.sums;
    num x, y, xy, x2, y2, k, a, b, c, s, px, py, ex, ey;
    bool r = false;

    if (j>=n) {
      j -= n;
      r = true;
    }

    if (!r) {
      x = sums[j+1].x - sums[i].x;
      y = sums[j+1].y - sums[i].y;
      x2 = sums[j+1].x2 - sums[i].x2;
      xy = sums[j+1].xy - sums[i].xy;
      y2 = sums[j+1].y2 - sums[i].y2;
      k = j+1 - i;
    } else {
      x = sums[j+1].x - sums[i].x + sums[n].x;
      y = sums[j+1].y - sums[i].y + sums[n].y;
      x2 = sums[j+1].x2 - sums[i].x2 + sums[n].x2;
      xy = sums[j+1].xy - sums[i].xy + sums[n].xy;
      y2 = sums[j+1].y2 - sums[i].y2 + sums[n].y2;
      k = j+1 - i + n;
    } 

    px = (pt[i].x + pt[j].x) / 2.0 - pt[0].x;
    py = (pt[i].y + pt[j].y) / 2.0 - pt[0].y;
    ey = (pt[j].x - pt[i].x);
    ex = -(pt[j].y - pt[i].y);

    a = ((x2 - 2*x*px) / k + px*px);
    b = ((xy - x*py - y*px) / k + px*py);
    c = ((y2 - 2*y*py) / k + py*py);

    s = ex*ex*a + 2*ex*ey*b + ey*ey*c;

    return Math.sqrt(s);
  }
  
  static void bestPolygon(_Path path) {
    int i, j, m, k, c, n = path.len;
    List<num> pen = List(n + 1);
    List<int> prev = List(n + 1), clip0 = List(n), clip1 = List(n + 1), seg0 = List(n + 1), seg1 = List(n + 1);
    num thispen, best;

    for (i=0; i<n; i++) {
      c = mod(path.lon[mod(i-1, n)]-1, n);
      if (c == i) {
        c = mod(i+1,n);
      }
      if (c < i) {
        clip0[i] = n;
      } else {
        clip0[i] = c;
      }
    }
    
    j = 1;
    for (i=0; i<n; i++) {
      while (j <= clip0[i]) {
        clip1[j] = i;
        j++;
      }
    }
    
    i = 0;
    for (j=0; i<n; j++) {
      seg0[j] = i;
      i = clip0[i];
    }
    seg0[j] = n;
    m = j;
  
    i = n;
    for (j=m; j>0; j--) {
      seg1[j] = i;
      i = clip1[i];
    }
    seg1[0] = 0;
    
    pen[0]=0;
    for (j=1; j<=m; j++) {
      for (i=seg1[j]; i<=seg0[j]; i++) {
        best = -1;
        for (k=seg0[j-1]; k>=clip1[i]; k--) {
          thispen = penalty3(path, k, i) + pen[k];
          if (best < 0 || thispen < best) {
            prev[i] = k;
            best = thispen;
          }
        }
        pen[i] = best;
      }
    }
    path.m = m;
    path.po = List(m);
 
    i = n;
    for (j=m-1; i>0; j--) {
      i = prev[i];
      path.po[j] = i;
    }
  }

  static void pointslope(_Path path, int i, int j, _DPoint ctr, _DPoint dir) {
    int n = path.len;
    List<_Sum> sums = path.sums;
    num x, y, x2, xy, y2, k, a, b, c, lambda2, l;
    int r=0;
  
    while (j>=n) {
      j-=n;
      r+=1;
    }
    while (i>=n) {
      i-=n;
      r-=1;
    }
    while (j<0) {
      j+=n;
      r-=1;
    }
    while (i<0) {
      i+=n;
      r+=1;
    }
    
    x = sums[j+1].x-sums[i].x+r*sums[n].x;
    y = sums[j+1].y-sums[i].y+r*sums[n].y;
    x2 = sums[j+1].x2-sums[i].x2+r*sums[n].x2;
    xy = sums[j+1].xy-sums[i].xy+r*sums[n].xy;
    y2 = sums[j+1].y2-sums[i].y2+r*sums[n].y2;
    k = j+1-i+r*n;
    
    ctr.x = x/k;
    ctr.y = y/k;
  
    a = (x2-x*x/k)/k;
    b = (xy-x*y/k)/k;
    c = (y2-y*y/k)/k;
    
    lambda2 = (a+c+Math.sqrt((a-c)*(a-c)+4*b*b))/2;
  
    a -= lambda2;
    c -= lambda2;
  
    if (a.abs() >= c.abs()) {
      l = Math.sqrt(a*a+b*b);
      if (l != 0) {
        dir.x = -b/l;
        dir.y = a/l;
      }
    } else {
      l = Math.sqrt(c*c+b*b);
      if (l != 0) {
        dir.x = -c/l;
        dir.y = b/l;
      }
    }
    if (l == 0) {
      dir.x = dir.y = 0; 
    }
  }
  
  static void adjustVertices(_Path path) {
    int n = path.len, m = path.m, x0 = path.x0, y0 = path.y0;
    List<int> po = path.po;
    List<_Point> pt = path.pt;
    List<_DPoint> ctr = List(m), dir = List(m);
    List<_Quad> q = List(m);
    List<num> v = List(3);
    int i, j, k, l;
    _DPoint s = _DPoint();
    num d;

    path.curve = _Curve(m);

    for (i=0; i<m; i++) {
      j = po[mod(i+1,m)];
      j = mod(j-po[i],n)+po[i];
      ctr[i] = _DPoint();
      dir[i] = _DPoint();
      pointslope(path, po[i], j, ctr[i], dir[i]);
    }
  
    for (i=0; i<m; i++) {
      q[i] = _Quad();
      d = dir[i].x * dir[i].x + dir[i].y * dir[i].y;
      if (d == 0.0) {
        for (j=0; j<3; j++) {
          for (k=0; k<3; k++) {
            q[i].data[j * 3 + k] = 0;
          }
        }
      } else {
        v[0] = dir[i].y;
        v[1] = -dir[i].x;
        v[2] = - v[1] * ctr[i].y - v[0] * ctr[i].x;
        for (l=0; l<3; l++) {
          for (k=0; k<3; k++) {
            q[i].data[l * 3 + k] = v[l] * v[k] / d;
          }
        }
      }
    }
  
    _Quad Q;
    _DPoint w;
    num dx, dy, det, min, cand, xmin, ymin;
    int z;
    for (i=0; i<m; i++) {
      Q = _Quad();
      w = _DPoint();
  
      s.x = (pt[po[i]].x-x0).toDouble();
      s.y = (pt[po[i]].y-y0).toDouble();
  
      j = mod(i-1,m);
      
      for (l=0; l<3; l++) {
        for (k=0; k<3; k++) {
          Q.data[l * 3 + k] = q[j].at(l, k) + q[i].at(l, k);
        }
      }
      
      while(true) {
        det = Q.at(0, 0)*Q.at(1, 1) - Q.at(0, 1)*Q.at(1, 0);
        if (det != 0.0) {
          w.x = (-Q.at(0, 2)*Q.at(1, 1) + Q.at(1, 2)*Q.at(0, 1)) / det;
          w.y = ( Q.at(0, 2)*Q.at(1, 0) - Q.at(1, 2)*Q.at(0, 0)) / det;
          break;
        }
  
        if (Q.at(0, 0)>Q.at(1, 1)) {
          v[0] = -Q.at(0, 1);
          v[1] = Q.at(0, 0);
        } else if (Q.at(1, 1) != 0) {
          v[0] = -Q.at(1, 1);
          v[1] = Q.at(1, 0);
        } else {
          v[0] = 1;
          v[1] = 0;
        }
        d = v[0] * v[0] + v[1] * v[1];
        v[2] = - v[1] * s.y - v[0] * s.x;
        for (l=0; l<3; l++) {
          for (k=0; k<3; k++) {
            Q.data[l * 3 + k] += v[l] * v[k] / d;
          }
        }
      }
      dx = (w.x-s.x).abs();
      dy = (w.y-s.y).abs();
      if (dx <= 0.5 && dy <= 0.5) {
        path.curve.vertex[i] = _DPoint(w.x+x0, w.y+y0);
        continue;
      }
  
      min = quadform(Q, s);
      xmin = s.x;
      ymin = s.y;
  
      if (Q.at(0, 0) != 0.0) {
        for (z=0; z<2; z++) {
          w.y = s.y-0.5+z;
          w.x = - (Q.at(0, 1) * w.y + Q.at(0, 2)) / Q.at(0, 0);
          dx = (w.x-s.x).abs();
          cand = quadform(Q, w);
          if (dx <= 0.5 && cand < min) {
            min = cand;
            xmin = w.x;
            ymin = w.y;
          }
        }
      }

      if (Q.at(1, 1) != 0.0) {
        for (z=0; z<2; z++) {
          w.x = s.x-0.5+z;
          w.y = - (Q.at(1, 0) * w.x + Q.at(1, 2)) / Q.at(1, 1);
          dy = (w.y-s.y).abs();
          cand = quadform(Q, w);
          if (dy <= 0.5 && cand < min) {
            min = cand;
            xmin = w.x;
            ymin = w.y;
          }
        }
      }

      for (l=0; l<2; l++) {
        for (k=0; k<2; k++) {
          w.x = s.x-0.5+l;
          w.y = s.y-0.5+k;
          cand = quadform(Q, w);
          if (cand < min) {
            min = cand;
            xmin = w.x;
            ymin = w.y;
          }
        }
      }

      path.curve.vertex[i] = _DPoint(xmin + x0, ymin + y0);
    }
  }
  
  static void reverse(_Path path) {
    _Curve curve = path.curve;
    int m = curve.n;
    List<_DPoint> v = curve.vertex;

    for (int i=0, j=m-1; i<j; i++, j--) {
      _DPoint tmp = v[i];
      v[i] = v[j];
      v[j] = tmp;
    }
  }

  void smooth(_Path path) {
    int m = path.curve.n;
    int i, j, k;
    num dd, denom, alpha;
    _DPoint p2, p3, p4;
    _Curve curve = path.curve;

    for (i=0; i<m; i++) {
      j = mod(i+1, m);
      k = mod(i+2, m);
      p4 = interval(1/2.0, curve.vertex[k], curve.vertex[j]);

      denom = ddenom(curve.vertex[i], curve.vertex[k]);
      if (denom != 0.0) {
        dd = dpara(curve.vertex[i], curve.vertex[j], curve.vertex[k]) / denom;
        dd = dd.abs();
        alpha = dd>1 ? (1 - 1.0/dd) : 0;
        alpha = alpha / 0.75;
      } else {
        alpha = 4/3.0;
      }
      curve.alpha0[j] = alpha;
  
      if (alpha >= alphamax) { 
        curve.tag[j] = "CORNER";
        curve.c[3 * j + 1] = curve.vertex[j];
        curve.c[3 * j + 2] = p4;
      } else {
        if (alpha < 0.55) {
          alpha = 0.55;
        } else if (alpha > 1) {
          alpha = 1;
        }
        p2 = interval(0.5+0.5*alpha, curve.vertex[i], curve.vertex[j]);
        p3 = interval(0.5+0.5*alpha, curve.vertex[k], curve.vertex[j]);
        curve.tag[j] = "CURVE";
        curve.c[3 * j + 0] = p2;
        curve.c[3 * j + 1] = p3;
        curve.c[3 * j + 2] = p4;
      }
      curve.alpha[j] = alpha;  
      curve.beta[j] = 0.5;
    }
    curve.alphacurve = true;
  }

  static bool optiPenalty(_Path path, int i, int j, _Opti res, num opttolerance, List<num> convc, List<num> areac) {
    int m = path.curve.n;
    int k, k1, k2, conv, i1;
    num area, alpha, d, d1, d2;
    _DPoint p0, p1, p2, p3, pt;
    num A, R, a1, a2, a3, a4, s, t;
    _Curve curve = path.curve;
    List<_DPoint> vertex = curve.vertex;

    if (i==j) {
      return true;
    }

    k = i;
    i1 = mod(i+1, m);
    k1 = mod(k+1, m);
    conv = convc[k1];
    if (conv == 0) {
      return true;
    }
    d = ddist(vertex[i], vertex[i1]);
    for (k=k1; k!=j; k=k1) {
      k1 = mod(k+1, m);
      k2 = mod(k+2, m);
      if (convc[k1] != conv) {
        return true;
      }
      if (sign(cprod(vertex[i], vertex[i1], vertex[k1], vertex[k2])) !=
          conv) {
        return true;
      }
      if (iprod1(vertex[i], vertex[i1], vertex[k1], vertex[k2]) <
          d * ddist(vertex[k1], vertex[k2]) * -0.999847695156) {
        return true;
      }
    }

    p0 = curve.c[mod(i,m) * 3 + 2].copy();
    p1 = vertex[mod(i+1,m)].copy();
    p2 = vertex[mod(j,m)].copy();
    p3 = curve.c[mod(j,m) * 3 + 2].copy();

    area = areac[j] - areac[i];
    area -= dpara(vertex[0], curve.c[i * 3 + 2], curve.c[j * 3 + 2])/2;
    if (i>=j) {
      area += areac[m];
    }

    a1 = dpara(p0, p1, p2);
    a2 = dpara(p0, p1, p3);
    a3 = dpara(p0, p2, p3);

    a4 = a1+a3-a2;    

    if (a2 == a1) {
      return true;
    }

    t = a3/(a3-a4);
    s = a2/(a2-a1);
    A = a2 * t / 2.0;

    if (A == 0.0) {
      return true;
    }

    R = area / A;
    alpha = 2 - Math.sqrt(4 - R / 0.3);

    res.c[0] = interval(t * alpha, p0, p1);
    res.c[1] = interval(s * alpha, p3, p2);
    res.alpha = alpha;
    res.t = t;
    res.s = s;

    p1 = res.c[0].copy();
    p2 = res.c[1].copy(); 

    res.pen = 0;

    for (k=mod(i+1,m); k!=j; k=k1) {
      k1 = mod(k+1,m);
      t = tangent(p0, p1, p2, p3, vertex[k], vertex[k1]);
      if (t<-0.5) {
        return true;
      }
      pt = bezier(t, p0, p1, p2, p3);
      d = ddist(vertex[k], vertex[k1]);
      if (d == 0.0) {
        return true;
      }
      d1 = dpara(vertex[k], vertex[k1], pt) / d;
      if (d1.abs() > opttolerance) {
        return true;
      }
      if (iprod(vertex[k], vertex[k1], pt) < 0 ||
          iprod(vertex[k1], vertex[k], pt) < 0) {
        return true;
      }
      res.pen += d1 * d1;
    }

    for (k=i; k!=j; k=k1) {
      k1 = mod(k+1,m);
      t = tangent(p0, p1, p2, p3, curve.c[k * 3 + 2], curve.c[k1 * 3 + 2]);
      if (t<-0.5) {
        return true;
      }
      pt = bezier(t, p0, p1, p2, p3);
      d = ddist(curve.c[k * 3 + 2], curve.c[k1 * 3 + 2]);
      if (d == 0.0) {
        return true;
      }
      d1 = dpara(curve.c[k * 3 + 2], curve.c[k1 * 3 + 2], pt) / d;
      d2 = dpara(curve.c[k * 3 + 2], curve.c[k1 * 3 + 2], vertex[k1]) / d;
      d2 *= 0.75 * curve.alpha[k1];
      if (d2 < 0) {
        d1 = -d1;
        d2 = -d2;
      }
      if (d1 < d2 - opttolerance) {
        return true;
      }
      if (d1 < d2) {
        res.pen += (d1 - d2) * (d1 - d2);
      }
    }

    return false;
  }

  void optiCurve(_Path path) { 
    _Curve curve = path.curve;
    int m = curve.n, i, j, om, i1;
    List<_DPoint> vert = curve.vertex; 
    List<int> pt = List(m + 1), len = List(m + 1), convc = List(m);
    List<num> pen = List(m + 1), areac = List(m + 1);
    List<_Opti> opt = List(m + 1);
    _Opti o = _Opti();
    _DPoint p0;
    num area, alpha;
    _Curve ocurve;

    for (i=0; i<m; i++) {
      if (curve.tag[i] == "CURVE") {
        convc[i] = sign(dpara(vert[mod(i-1,m)], vert[i], vert[mod(i+1,m)]));
      } else {
        convc[i] = 0;
      }
    }
  
    area = 0.0;
    areac[0] = 0.0;
    p0 = curve.vertex[0];
    for (i=0; i<m; i++) {
      i1 = mod(i+1, m);
      if (curve.tag[i1] == "CURVE") {
        alpha = curve.alpha[i1];
        area += 0.3 * alpha * (4-alpha) *
            dpara(curve.c[i * 3 + 2], vert[i1], curve.c[i1 * 3 + 2])/2;
        area += dpara(p0, curve.c[i * 3 + 2], curve.c[i1 * 3 + 2])/2;
      }
      areac[i+1] = area;
    }
  
    pt[0] = -1;
    pen[0] = 0;
    len[0] = 0; 
  
    for (j=1; j<=m; j++) {
      pt[j] = j-1;
      pen[j] = pen[j-1];
      len[j] = len[j-1]+1;
  
      for (i=j-2; i>=0; i--) {
        bool r = optiPenalty(path, i, mod(j,m), o, opttolerance, convc, 
            areac);
        if (r) {
          break;
        }
          if (len[j] > len[i]+1 ||
              (len[j] == len[i]+1 && pen[j] > pen[i] + o.pen)) {
            pt[j] = i;
            pen[j] = pen[i] + o.pen;
            len[j] = len[i] + 1;
            opt[j] = o;
            o = _Opti();
          }
      }
    }
    om = len[m];
    ocurve = _Curve(om);
    List<num> s = List(om), t = List(om);
  
    j = m;
    for (i=om-1; i>=0; i--) {
      if (pt[j]==j-1) {
        ocurve.tag[i]     = curve.tag[mod(j,m)];
        ocurve.c[i * 3 + 0]    = curve.c[mod(j,m) * 3 + 0];
        ocurve.c[i * 3 + 1]    = curve.c[mod(j,m) * 3 + 1];
        ocurve.c[i * 3 + 2]    = curve.c[mod(j,m) * 3 + 2];
        ocurve.vertex[i]  = curve.vertex[mod(j,m)];
        ocurve.alpha[i]   = curve.alpha[mod(j,m)];
        ocurve.alpha0[i]  = curve.alpha0[mod(j,m)];
        ocurve.beta[i]    = curve.beta[mod(j,m)];
        s[i] = t[i] = 1.0;
      } else {
        ocurve.tag[i] = "CURVE";
        ocurve.c[i * 3 + 0] = opt[j].c[0];
        ocurve.c[i * 3 + 1] = opt[j].c[1];
        ocurve.c[i * 3 + 2] = curve.c[mod(j,m) * 3 + 2];
        ocurve.vertex[i] = interval(opt[j].s, curve.c[mod(j,m) * 3 + 2],
                                     vert[mod(j,m)]);
        ocurve.alpha[i] = opt[j].alpha;
        ocurve.alpha0[i] = opt[j].alpha;
        s[i] = opt[j].s;
        t[i] = opt[j].t;
      }
      j = pt[j];
    }
  
    for (i=0; i<om; i++) {
      i1 = mod(i+1,om);
      ocurve.beta[i] = s[i] / (s[i] + t[i1]);
    }
    ocurve.alphacurve = true;
    path.curve = ocurve;
  }
 
  void run(List<_Path> pathlist) {
    for (int i = 0; i < pathlist.length; i++) {
      _Path path = pathlist[i];
      calcSums(path);
      calcLon(path);
      bestPolygon(path);
      adjustVertices(path);

      if (path.sign == "-")
        reverse(path);

      smooth(path);

      if (optcurve)
        optiCurve(path);
    }
  }
}

class _GetSVG {
  _Bitmap bm;
  num size;
  String optType;
  List<_Path> pathlist;
  _GetSVG(this.bm, this.size, this.optType, this.pathlist);

  String bezier(_Curve curve, int i) {
    return 'C '
      + (curve.c[i * 3 + 0].x * size).toStringAsFixed(3) + ' '
      + (curve.c[i * 3 + 0].y * size).toStringAsFixed(3) + ','
      + (curve.c[i * 3 + 1].x * size).toStringAsFixed(3) + ' '
      + (curve.c[i * 3 + 1].y * size).toStringAsFixed(3) + ','
      + (curve.c[i * 3 + 2].x * size).toStringAsFixed(3) + ' '
      + (curve.c[i * 3 + 2].y * size).toStringAsFixed(3) + ' ';
  }

  String segment(_Curve curve, int i) {
    return 'L '
      + (curve.c[i * 3 + 1].x * size).toStringAsFixed(3) + ' '
      + (curve.c[i * 3 + 1].y * size).toStringAsFixed(3) + ' '
      + (curve.c[i * 3 + 2].x * size).toStringAsFixed(3) + ' '
      + (curve.c[i * 3 + 2].y * size).toStringAsFixed(3) + ' ';
  }

  String path(_Curve curve) {
    int n = curve.n;
    String p = 'M'
      + (curve.c[(n - 1) * 3 + 2].x * size).toStringAsFixed(3) + ' '
      + (curve.c[(n - 1) * 3 + 2].y * size).toStringAsFixed(3) + ' ';

    for (int i = 0; i < n; i++) {
      switch(curve.tag[i]) {
        case 'CURVE':
          p += bezier(curve, i);
          break;

        case 'CORNER':
          p += segment(curve, i);
          break;
      }
    }
    return p;
  }

  String run() {
    num w = bm.w * size, h = bm.h * size;
    String svg = '<svg id="svg" version="1.1" width="' + w.toString() + '" height="' + h.toString() + '" xmlns="http://www.w3.org/2000/svg">';

    svg += '<path d="';
    for (int i = 0; i < pathlist.length; i++)
      svg += path(pathlist[i].curve);

    String strokec, fillc, fillrule;
    if (optType == "curve") {
      strokec = "black";
      fillc = "none";
      fillrule = '';
    } else {
      strokec = "none";
      fillc = "black";
      fillrule = ' fill-rule="evenodd"';
    }

    svg += '" stroke="' + strokec + '" fill="' + fillc + '"' + fillrule + '/></svg>';
    return svg;
  }
}

class _GetPath {
  _Bitmap bm;
  num size;
  String optType;
  List<_Path> pathlist;
  Path ret = Path();
  _GetPath(this.bm, this.size, this.optType, this.pathlist);

  void path(_Curve curve) {
    int n = curve.n;
    ret.moveTo(
      curve.c[(n - 1) * 3 + 2].x * size,
      curve.c[(n - 1) * 3 + 2].y * size);

    for (int i = 0; i < n; i++) {
      switch(curve.tag[i]) {
        case 'CURVE':
          ret.cubicTo(
            curve.c[i * 3 + 0].x * size,
            curve.c[i * 3 + 0].y * size,
            curve.c[i * 3 + 1].x * size,
            curve.c[i * 3 + 1].y * size,
            curve.c[i * 3 + 2].x * size,
            curve.c[i * 3 + 2].y * size
          );
          break;

        case 'CORNER':
          ret.lineTo(
            curve.c[i * 3 + 1].x * size,
            curve.c[i * 3 + 1].y * size
          );
          ret.lineTo(
            curve.c[i * 3 + 2].x * size,
            curve.c[i * 3 + 2].y * size
          );
          break;
      }
    }
  }

  Path run() {
    for (int i = 0; i < pathlist.length; i++) 
      path(pathlist[i].curve);
    return ret;
  }
}
