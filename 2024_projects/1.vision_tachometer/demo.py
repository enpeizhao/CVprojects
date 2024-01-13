'''
@Author: enpeicv
@Date: 2024-01-10
vision based tachometer
'''
import cv2
import numpy as np
import time
import argparse

class VideoProcessor:
    def __init__(self, video_path, snap_path, output_path='result.mp4'):
        # read the video
        if video_path == '0':
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(video_path)
        # set video resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)

        # get video info
        self.cap_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print("Height: {}, Width: {}, FPS: {}".format(self.cap_h, self.cap_w, self.fps) )
        # create video writer
        self.writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (self.cap_w//2, self.cap_h//2))
        # read the reference image
        self.img = cv2.imread(snap_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # self.img = cv2.resize(self.img, (0, 0), fx=0.5, fy=0.5)

        self.data_list = []
        
    
    def calculate_similarity(self, image1, image2):
        '''
        use SIFT to calculate similarity
        @param image1: reference image
        @param image2: current image

        @return similarity: similarity between image1 and image2
        '''
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

        flann = cv2.FlannBasedMatcher()
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        similarity = len(good_matches) / max(len(keypoints1), len(keypoints2))
        
        match_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)

        return similarity, match_image
    
    def find_peaks(self, data, threshold):
        '''
        find peaks of data series
        @param data: data series
        @param threshold: threshold of peaks

        @return peaks_index: index of peaks
        '''
        peaks_index = []
        for i in range(2, len(data)-2):
            if data[i] > data[i-1] and data[i] > data[i-2] and data[i] > data[i+1] and data[i] > data[i+2] and data[i] > threshold:
                peaks_index.append(i)
        return peaks_index

    def draw_graph(self, bg, data_list, color=(0, 255, 0)):
        '''
        draw similarity graph, display rotation speed
        @param bg: background image
        @param data_list: similarity data series
        @param color: color of graph

        @return find_new_revolution: if find new revolution (True/False)
        '''

        find_new_revolution = False
        
        # draw coordinate axis
        line1_left_p = (0,400)
        line1_right_p = (800,400)

        line2_left_p = (400,0)
        line2_right_p = (400,800)
        cv2.line(bg,line1_left_p,line1_right_p,(0,255,0),1)
        cv2.line(bg,line2_left_p,line2_right_p,(0,255,0),1)

        # draw graph
        all_points = []
        for i,y_value in enumerate( data_list[-1000:][::-1] ): # ::-1 reverse list
            x = 700 + (-3 * i) 
            y = 400 - (300 * y_value)
            y = int(y)

            p = (x,y)

            # draw the first point
            if x == 700:
                cv2.circle(bg,(x,y),10,(255,0,255),-1)
                cv2.putText(bg, str(round(y_value,3)), (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            all_points.append(p)
        # draw the polyline
        pts = np.asarray(all_points,np.int32)
        pts = pts.reshape((-1, 1, 2)) 
        cv2.polylines(bg, [pts], False, color, 2) 

        # find peaks
        peaks_index = self.find_peaks(data_list[-1000:][::-1], 0.2)
        # draw peaks
        for i in peaks_index:
            x = 700 + (-3 * i)
            if i == 2:
                # if new revolution, find_new_revolution = True
                find_new_revolution = True

            y_value = data_list[-1000:][::-1][i]
            y = 400 - (300 * y_value)
            y = int(y)
            cv2.circle(bg,(x,y),5,(0,255,0),-1)
            cv2.putText(bg, str(round(y_value,3)), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

       
        return find_new_revolution

    def process_video(self):
        last_peak_time = time.time()
        last_rpm = 0
        while True:
            # create background image
            bg = np.zeros((800, 800, 3), np.uint8)
            rpm_color = (0, 255, 0)
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)


            similarity, new_img = self.calculate_similarity( self.img, gray)
            self.data_list.append(similarity)

            find_new_revolution = self.draw_graph(bg, self.data_list, (255, 0, 255))

            # check if find new revolution
            if find_new_revolution == True:
                # calculate rotation speed RPM(Revolutions Per Minute)
                last_rpm = 60 / (time.time() - last_peak_time)
                last_peak_time = time.time()
                rpm_color = (0, 0, 255)
                
            
            end_time = time.time()
            duration = end_time - start_time
            fps = 1 / duration
            # fps
            cv2.putText(bg, "fps: " + str(round(fps, 2)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # rpm
            cv2.putText(bg, "rpm: " + str(round(last_rpm, 1)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, rpm_color, 2)


            cv2.imshow('frame', gray)
            cv2.imshow('bg', bg)
            cv2.imshow('new_img', new_img)

            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.writer.write(gray)

            if cv2.waitKey(1) == ord('q'):
                break


        # Release resources
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='media/1.video_tachometer_demo.mp4', help='path to video')
    parser.add_argument('--snap_path', type=str, default='media/snap.png', help='path to snap')
    parser.add_argument('--output_path', type=str, default='result.mp4', help='path to output video')
    args = parser.parse_args()
    # usage: python demo.py --video_path media/1.video_tachometer_demo.mp4 --snap_path media/snap.png --output_path result.mp4

    video_processor = VideoProcessor(args.video_path, args.snap_path, args.output_path)
    video_processor.process_video()
