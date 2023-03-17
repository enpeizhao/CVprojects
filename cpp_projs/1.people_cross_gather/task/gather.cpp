#include "gather.h"

#include <cmath>
#include <stdio.h>

using namespace std;

// function for k-means alogrithm
double distance(const Point& p1, const Point& p2) {
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

vector<vector<Point>> kMeans(const vector<Point>& points, int k, int maxIterations) {
    int n = points.size();
    vector<Point> centroids(k);
    vector<vector<Point>> clusters(k);
    for (int i = 0; i < k; i++) {
        centroids[i] = points[rand() % n];
    }
    for (int iter = 0; iter < maxIterations; iter++) {
        for (int i = 0; i < k; i++) {
            clusters[i].clear();
        }
        for (int i = 0; i < n; i++) {
            double minDist = distance(points[i], centroids[0]);
            int minIndex = 0;
            for (int j = 1; j < k; j++) {
                double d = distance(points[i], centroids[j]);
                if (d < minDist) {
                    minDist = d;
                    minIndex = j;
                }
            }
            clusters[minIndex].push_back(points[i]);
        }
        for (int i = 0; i < k; i++) {
            double sumX = 0, sumY = 0;
            int m = clusters[i].size();
            if (m == 0) {continue;}
            for (int j = 0; j < m; j++) {
                sumX += clusters[i][j].x;
                sumY += clusters[i][j].y;
            }
            centroids[i].x = sumX / m;
            centroids[i].y = sumY / m;
        }
    }
    return clusters;
}

float getStdDev(const vector<Point>& points) {
    float sumX = 0, sumY = 0;
    int n = points.size();
    for (int i = 0; i < n; i++) {
        sumX += points[i].x;
        sumY += points[i].y;
    }
    float meanX = sumX / n;
    float meanY = sumY / n;
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += pow(points[i].x - meanX, 2) + pow(points[i].y - meanY, 2);
    }
    return sqrt(sum / n);
}

void isGather(const vector<vector<Point>>& clusters, float threshold, vector<vector<Point>>& gatherPoints) {
    gatherPoints.clear();
    int k = clusters.size();
    for (int i = 0; i < k; i++) {
        if (clusters[i].size() == 0) {
            continue;
        }
        auto std = getStdDev(clusters[i]);
        //printf("std: %f\n", std);
        if (std < threshold && clusters[i].size() > 2) {
            gatherPoints.push_back(clusters[i]);
        }
    }
}

vector<vector<Point>> gather(const vector<Point>& points) {
    vector<vector<Point>> clusters = kMeans(points, 10, 100);
    vector<vector<Point>> gatherPoints;
    isGather(clusters, 80, gatherPoints);
    return gatherPoints;
}

Point averagePoint(const vector<Point>& points) {
    float sumX = 0, sumY = 0;
    int n = points.size();
    for (int i = 0; i < n; i++) {
        sumX += points[i].x;
        sumY += points[i].y;
    }
    return Point{int(sumX / n), int(sumY / n)};
}

std::vector<std::vector<Point>> gather_rule(const std::vector<Point>& points, float threshold) {
    // float threshold = 200;
    std::vector<std::vector<Point>> gatherPoints;
    for (int i = 0; i < points.size(); i++) {
       for(auto& pts : gatherPoints) {
            float dist = distance(points[i], averagePoint(pts));
            // printf("dist: %f\n", dist);
            if ( dist < threshold) {
                pts.push_back(points[i]);
                break;
            }
       } 
       gatherPoints.push_back(std::vector<Point>{points[i]});
    }
    return gatherPoints;
}