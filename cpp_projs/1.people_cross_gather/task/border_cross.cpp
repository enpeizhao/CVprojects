#include "border_cross.h"

// explain this function
// https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
bool onSegment(Point p, Point q, Point r) {
    if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) && q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y)) {
        return true;
    }
    return false;
}

// https://www.geeksforgeeks.org/orientation-3-ordered-points/
int orientation(Point p, Point q, Point r) {
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0) {
        return 0;
    }
    return (val > 0) ? 1 : 2;
}

bool doIntersect(Point p1, Point q1, Point p2, Point q2) {
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);
    if (o1 != o2 && o3 != o4 && o4 != 0) {
        return true;
    }
    if (o1 == 0 && onSegment(p1, p2, q1)) {
        return true;
    }
    if (o2 == 0 && onSegment(p1, q2, q1)) {
        return true;
    }
    if (o3 == 0 && onSegment(p2, p1, q2)) {
        return true;
    }
    if (o4 == 0 && onSegment(p2, q1, q2)) {
        return false;
    }
    return false;
}

// Function that checks if the point is inside the polygon
bool isInside(Polygon polygon, Point p) {
    int n = polygon.size();
    if (n < 3) {
        return false;
    }
    Point extreme = {10000, p.y};
    int count = 0, i = 0;
    do {
        int next = (i + 1) % n;
        if (doIntersect(polygon[i], polygon[next], p, extreme)) {
            if (orientation(polygon[i], p, polygon[next]) == 0) {
                return onSegment(polygon[i], p, polygon[next]);
            }
            count++;
        }
        i = next;
    } while (i != 0);
    return count % 2 == 1;
}