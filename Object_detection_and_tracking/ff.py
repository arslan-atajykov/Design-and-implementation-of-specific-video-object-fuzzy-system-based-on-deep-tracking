
import sys

import os


# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS

import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

from PyQt5.QtWidgets import QFileDialog,QApplication,QSlider, QLabel, QMainWindow,QPushButton, QLineEdit,QGridLayout,QMessageBox,QWidget,QPlainTextEdit,QVBoxLayout, QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QProcess,QThread, pyqtSignal,QObject
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *







class UIWindow(QWidget):
    def __init__(self, parent=None):
        super(UIWindow, self).__init__(parent)

        self.p = None
        self.mohu_size=0
        self.class_ob='person'

        self.path=''
        self.layout = QGridLayout()
        self.class_label = QLabel("特定目标")
        self.blur_label = QLabel("模糊率")
        self.start_btn = QPushButton("开始")
        self.object_select = QComboBox()
        self.object_select.addItem("person")
        self.object_select.addItem("car")
        self.object_select.addItem("bicycle")
        self.object_select.addItem("motorbike")
        self.object_select.addItem("aeroplane")
        self.object_select.addItem("bus")
        self.object_select.addItem("train")
        self.object_select.addItem("truck")
        self.object_select.addItem("boat")

        self.object_select.addItem("bird")
        self.object_select.addItem("cat")
        self.object_select.addItem("dog")
        self.object_select.addItem("horse")
        self.object_select.addItem("sheep")
        self.object_select.addItem("cow")
        self.object_select.addItem("elephant")
        self.object_select.addItem("bear")


        self.object_select.currentIndexChanged.connect(self.selectionchange)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(5)
        self.slider.setMaximum(45)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)

        self.slider.valueChanged.connect(self.valuechange)


        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.layout.addWidget(QLabel(""), 0, 0)
        self.layout.addWidget(QLabel(""), 1, 0)
        self.layout.addWidget(QLabel(""), 2, 0)
        self.layout.addWidget(QLabel(""), 3, 0)
        self.layout.addWidget(QLabel(""), 4, 0)
        self.layout.addWidget(QLabel(""), 5, 0)

        self.layout.addWidget(QLabel(""), 0, 1)
        self.layout.addWidget(QLabel(""), 0, 2)
        self.layout.addWidget(QLabel(""), 0, 3)
        self.layout.addWidget(QLabel(""), 0, 4)
        self.start_btn.setStyleSheet(stylesheet)
        self.start_btn.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.layout.addWidget(self.class_label,1,1)
        self.layout.addWidget(self.object_select,2,1)
        self.layout.addWidget(self.blur_label,1,3)
        self.layout.addWidget(self.slider,2,3)
        self.layout.addWidget(self.start_btn,4,3)
        self.start_btn.clicked.connect(self.start_tracker)

        self.setLayout(self.layout)

    def valuechange(self):
      size = self.slider.value()
      self.mohu_size = size
      print("size"+ str(size))


    def selectionchange(self,i):
       print(self.path)
       # print ("Items in the list are :")



       # for count in range(self.object_select.count()):
       #    print (self.object_select.itemText(count))
       print ("Current index",i,"selection changed ",self.object_select.currentText())
       self.class_ob=self.object_select.currentText()


    def start_tracker(self):
        weights='./checkpoints/yolov4-416'
        video_path = self.path
        out = "/Users/macbookpro/Desktop/毕设/finished/test01.avi"
        ob_count = 'False'







        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0

        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = 416



        # load tflite model if flag is set

        # interpreter = tf.lite.Interpreter(model_path=weights)
        # interpreter.allocate_tensors()
        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()
        # print(input_details)
        # print(output_details)
        # otherwise load standard tensorflow saved model

        saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

        # begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)



        # get video ready to save locally if flag is set

            # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out, codec, fps, (width, height))

        frame_num = 0
        # while video is running
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has ended !')
                break
            frame_num +=1
            print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            # run detections on tflite if flag is set

            # interpreter.set_tensor(input_details[0]['index'], image_data)
            # interpreter.invoke()
            # pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            #     # run detections using yolov3 if flag is set
            #
            #
            # boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
            #                                         input_shape=tf.constant([input_size, input_size]))

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.50
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())

            # custom allowed classes (uncomment line below to customize tracker for only people)
            # allowed_classes = ['person','bicycle','car','bus','bus','motorbike']
            allowed_classes = [self.class_ob]
            cv2.putText(frame, "specific class: {}".format(allowed_classes), (5, 65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            if ob_count:
                cv2.putText(frame, "currently tracking: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)

            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)


            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()

            # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                # frame[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])] = cv2.GaussianBlur(frame[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])],(45,45),0)
                try:

                    frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] = cv2.GaussianBlur(frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])],(self.mohu_size,self.mohu_size),0)
                except:
                    print("err")
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                # cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # if enable info flag then print details about each track
                # if FLAGS.info:
                #     print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

            # if output flag is set, save video file

            out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()






class UIToolTab(QWidget):
    def __init__(self, parent=None):
        super(UIToolTab, self).__init__(parent)

        self.second_window = UIWindow()
        # print(self.second_window.blur_label.text())

        self.layout = QGridLayout()
        self.name_label = QLabel("基于深度跟踪的视频特定目标模糊系统")
        self.upload_label = QLabel("上传并分析")
        self.ready_btn = QPushButton("next")

        self.upload_btn = QPushButton("upload")
        self.upload_btn.clicked.connect(self.open_file)

        self.upload_btn.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.ready_btn.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        # self.upload_btn.setGeometry(QtCore.QRect(102, 90, 151, 32))

        self.history_btn = QPushButton("history")
        self.history_btn.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.history_btn.clicked.connect(self.open_history_file)

        # self.Worker1 = Worker1()

        self.upload_label.setStyleSheet('font-family: "Arial";color: "#630660";font-size: 25px;')

        self.name_label.setStyleSheet('font-family: "Arial";color: blue;font-size: 32px; margin-bottom:15px')
        # self.upload_btn.setGeometry(102, 90, 151, 32)


        self.layout.addWidget(self.name_label, 0, 1,1,3)
        self.layout.addWidget(QLabel(""), 1, 0)
        self.layout.addWidget(QLabel(""), 2, 0)
        self.layout.addWidget(QLabel(""), 3, 0)
        self.layout.addWidget(QLabel(""), 4, 0)
        self.layout.addWidget(QLabel(""), 5, 0)

        self.layout.addWidget(QLabel(""), 0, 1)
        self.layout.addWidget(QLabel(""), 0, 2)
        self.layout.addWidget(QLabel(""), 0, 3)
        self.layout.addWidget(QLabel(""), 0, 4)

        self.layout.addWidget(self.upload_label,1,1)
        self.layout.addWidget(self.upload_btn,2,1)
        self.layout.addWidget(self.history_btn,2,3)
        self.layout.addWidget(self.ready_btn,5,2)
        self.setLayout(self.layout)


    def open_history_file(self):
        file_filter = 'Data File (*.mp4 *.avi)'
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption = 'Select a data file',
            directory = "/Users/macbookpro/Desktop/毕设/finished/",
            filter = file_filter,

        )
        self.video_path2 = response[0]
        cap = cv2.VideoCapture(self.video_path2)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        # self.Worker1.st = self.video_path2
        # self.Worker1.start()
        # self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        cv2.destroyAllWindows()



    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def open_file(self):

        file_filter = 'Data File (*.mp4 *.avi)'
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption = 'Select a data file',
            # directory = os.getcwd(),
            directory = "/Users/macbookpro/Desktop",
            filter = file_filter,

        )
        self.video_path = response[0]


        # print(self.video_path)

        # self.second_window.class_label.setText(self.video_path)
        #response='/Users/macbookpro/Desktop/git_deep/yolov4/data/video/mot.mp4', 'Data File (*.mp4 *.avi)'
        #response[0]=/Users/macbookpro/Desktop/git_deep/yolov4/data/video/test.mp4
        #print(response[0])

        # QString fileName = QFileDialog::getOpenFileName(this,tr("Select video"),"Users/macbookpro/Desktop/git_deep/yolov4/data/video",tr("Video files (*.mp4)"));



        # self.layout.setRowMinimumHeight(5,75)

        # self.button_Start = QPushButton('go to main window')


        # self.layout.addWidget(self.button_Start,5,1)

    # def passingInformation(self):
    #     self.


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self):
        self.Window = UIToolTab(self)
        self.ThreadActive = True
        # cap = cv2.VideoCapture(self.Window.video_path2)
        cap = cv2.VideoCapture(st)
        while self.ThreadActive:
            ret, frame = cap.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)

        cap.release()
    def stop(self):
        self.ThreadActive = False
        self.quit()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setGeometry(50, 50, 400, 450)
        # self.setFixedSize(800, 600)
        # self.layout = QGridLayout()
        # self.welcome_label = QLabel('Welcome')
        # self.layout.addWidget(self.welcome_label,2,2)
        self.setObjectName('MainWindow')

        # time_duration = 10
        # time.sleep(time_duration)
        stylesheet = '''
    #MainWindow {


        background-image: url(back.png);
        background-position: center;

        background-repeat: no-repeat;



    }
'''
        self.setStyleSheet(stylesheet)
        self.startUIToolTab()

    def startUIToolTab(self):
        self.ToolTab = UIToolTab(self)
        self.setWindowTitle("main_tab")
        self.setCentralWidget(self.ToolTab)
        self.ToolTab.setStyleSheet(stylesheet)

        self.ToolTab.upload_btn.clicked.connect(self.startUIWindow)

        self.ToolTab.ready_btn.clicked.connect(self.showUIWindow)

        self.show()


    def startUIWindow(self):
        self.Window = UIWindow(self)

    def showUIWindow(self):


        # self.Window.blur_label.setText(self.ToolTab.video_path)
        self.Window.path=self.ToolTab.video_path
        self.setCentralWidget(self.Window)

        self.show()



stylesheet="""

QPushButton{



    border: 2px solid red;
    font-size: 25px;
    text-align: center;
    border-radius: 14px;
    padding: 15px 5px;
    margin: 10px 30px 30px 30px;
    }



QPushButton:hover{
    background-color: #7eb5ed;}
}
"""



if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setStyleSheet(stylesheet)
    w = MainWindow()
    # w.setStyleSheet("background: 'back.png';")
    sys.exit(app.exec_())





# QPushButton{


#     color: black;
#     border: 4px solid red;
#     font-size: 30px;
#     text-align: center;
#     border-radius: 10px;
#     padding:5px 5px;
#     margin: 10px 30px 30px 30px;
#     *:hover{background: 'BC006C';}
# }


# self.upload_btn.setStyleSheet(
        #     """{border: 4px solid black;
        #        color: white;
        #        font-size: 16px;
        #        border-radius: 25px;
        #        padding: 15px 0;}
        #        """
        #     )
        # self.upload_btn.setStyleSheet('border: 4px solid red;'


        #         'color: black;'

        #         'font-size: 30px;'
        #         'border-radius: 10px;'
        #         'padding: 5px 5px;'
        #         'margin: 10px 30px 30px 30px;'

        #     )
        # self.history_btn.setStyleSheet('border: 4px solid red;'


        #         'color: black;'

        #         'font-size: 30px;'
        #         'border-radius: 10px;'
        #         'padding:5px 5px;'
        #         'margin: 10px 30px 30px 30px;'

        #     )
        # self.ready_btn.setStyleSheet('border: 4px solid red;'


        #         'color: black;'

        #         'font-size: 30px;'
        #         'text-align: center;'
        #         'border-radius: 10px;'
        #         'padding:5px 5px;'
        #         'margin: 10px 30px 30px 30px;'

        #     )
