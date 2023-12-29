import sys
import os
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog,QSlider, QLabel, QMainWindow,QPushButton, QLineEdit,QGridLayout,QMessageBox,QWidget,QPlainTextEdit,QVBoxLayout, QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QProcess,QThread, pyqtSignal,QObject
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
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

import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import cv2
import numpy as np




class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("main_dialog.ui",self)
        self.second_w = SecondUI()
        self.third_w = ThirdUI()
        # self.upload_btn.setIcon(QIcon('up_icon.png'))
        self.upload_btn.clicked.connect(self.upload)
        self.history_btn.clicked.connect(self.open_history_file)
        self.help_btn.clicked.connect(self.help)
        self.next_btn.clicked.connect(self.goToNext)
        self.roi_select.clicked.connect(self.goToThird)
        self.exit_btn.clicked.connect(self.exitt)
        # self.login.clicked.connect(self.gotologin)
        # self.create.clicked.connect(self.gotocreate)




    def open_history_file(self):
        file_filter = 'Data File (*.mp4 *.avi)'
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption = 'Select a data file',
            directory = "/Users/macbookpro/Desktop/毕设/yolov4/analyzed_vids/finished",
            filter = file_filter,

        )
        self.video_path2 = response[0]

    def upload(self):
        file_filter = 'Data File (*.mp4 *.avi)'
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption = 'Select a data file',
            # directory = os.getcwd(),
            directory = "/Users/macbookpro/Desktop",
            filter = file_filter,

        )
        self.video_path = response[0]
        self.label_2.setText("The video file {} has been succesfully uploaded \n\n\nClick target blur to continue or roi select".format(response[0]))

        

    def goToNext(self):
        self.second_w.path=self.video_path
        widget.addWidget(self.second_w)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def goToThird(self):
        self.third_w.video_path=self.video_path
        widget.addWidget(self.third_w)
        widget.setCurrentIndex(widget.currentIndex()+1)


    def help(self):

        self.label_2.setText("help meee")

    def exitt(self):
        sys.exit()

    
    # def gotologin(self):
    #     login = LoginScreen()
    #     widget.addWidget(login)
    #     widget.setCurrentIndex(widget.currentIndex()+1)

    # def gotocreate(self):
    #     create = CreateAccScreen()
    #     widget.addWidget(create)
    #     widget.setCurrentIndex(widget.currentIndex() + 1)

class SecondUI(QDialog):
    def __init__(self):
        super(SecondUI, self).__init__()
        loadUi("diaolog2.ui",self)

        self.path=''
        self.p = None
        self.mohu_size=0
        self.class_ob='person'
        self.class_exception=[]
        self.exception=False
        self.all_classes=['person','car','bicycle','motorbike','aeroplane','bus','train','truck','boat','bird','cat','dog','horse','sheep','cow','elephant','bear']

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


        self.slider.setMinimum(5)
        self.slider.setMaximum(45)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)


        self.object_select.currentIndexChanged.connect(self.selectionchange)
        self.slider.valueChanged.connect(self.valuechange)
        self.start_btn.clicked.connect(self.start_tracker)


    def valuechange(self):
      size = self.slider.value()
      self.mohu_size = size
      print("size"+ str(size))
      print(self.class_exception)
      if(self.exception):
        output = [x for x in self.all_classes if not x in self.class_exception or self.class_exception.remove(x)]
        self.class_ob=list(output)
        print(self.class_ob)


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
        out = './outputs/'+self.save_name.toPlainText()+'.avi'
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
        out = cv2.VideoWriter('./outputs/test02.avi', codec, fps, (width, height))

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
            if(self.exception):
                allowed_classes=self.class_ob
            else:
                allowed_classes = [self.class_ob]
                cv2.putText(frame, "blurring objects are : {}".format(allowed_classes), (5, 65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            
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
            # if ob_count:
            #     cv2.putText(frame, "blurred objects count: {}".format(count), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            #     print("Objects being tracked: {}".format(count))
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
        

        
        
class ThirdUI(QDialog):
    def __init__(self):
        super(ThirdUI, self).__init__()
        loadUi("diaolog3.ui",self)
        self.video_path= ''
        self.Worker1 = Worker1()
        self.second_w = SecondUI()
        
        

        
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.play_btn.clicked.connect(self.playy)
        self.pause_btn.clicked.connect(self.CancelFeed)
        self.analyze_btn.clicked.connect(self.goToSecond)

    def goToSecond(self):
        self.second_w.path=self.video_path
        self.second_w.object_select.hide()
        self.second_w.label_2.hide()
        self.second_w.class_exception=list(set(self.Worker1.cl))
        self.second_w.exception=True
        widget.addWidget(self.second_w)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
        print(self.Worker1.cl)
        



    def playy(self):
        self.Worker1.video_path=self.video_path
        print("worker {}".format(self.Worker1.video_path))
        self.Worker1.start()

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()



def get_output_layers(net):
    
        layer_names = net.getLayerNames()
        
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

class Worker1(QThread):
    # self.second_w = SecondUI()
    cl=[]
    video_path=''
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(self.video_path)
        while self.ThreadActive:
            ret, self.frame = Capture.read()
            if ret:
                rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1500, 550, Qt.KeepAspectRatio)
               
                self.ImageUpdate.emit(p)
            else :
                break;

             
     

    def stop(self):
        self.ThreadActive = False
        bbox = cv2.selectROI(self.frame, False)
        print("{}".format((int(bbox[0]),int(bbox[1]), int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))))
        cropped_image = self.frame[int(bbox[1]):int(bbox[1]+bbox[3]), 
                      int(bbox[0]):int(bbox[0]+bbox[2])]
        image = cropped_image
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        classes = None


       

        cl = []


        with open("yolov3.txt", 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])



        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            self.cl.append(str(classes[class_ids[i]]))

        print(self.cl)
       
        # cv2.imwrite("object-detection2.jpg", image)
        cv2.destroyAllWindows()
        # self.second_w.path=self.video_path
        # self.second_w.object_select.hide()

        
        
        self.quit()









# class LoginScreen(QDialog):
#     def __init__(self):
#         super(LoginScreen, self).__init__()
#         loadUi("main_interface.ui",self)
    #     self.passwordfield.setEchoMode(QtWidgets.QLineEdit.Password)
    #     self.login.clicked.connect(self.loginfunction)

    # def loginfunction(self):
    #     user = self.emailfield.text()
    #     password = self.passwordfield.text()

    #     if len(user)==0 or len(password)==0:
    #         self.error.setText("Please input all fields.")

    #     else:
    #         conn = sqlite3.connect("shop_data.db")
    #         cur = conn.cursor()
    #         query = 'SELECT password FROM login_info WHERE username =\''+user+"\'"
    #         cur.execute(query)
    #         result_pass = cur.fetchone()[0]
    #         if result_pass == password:
    #             print("Successfully logged in.")
    #             self.error.setText("")
    #         else:
    #             self.error.setText("Invalid username or password")

# class CreateAccScreen(QDialog):
#     def __init__(self):
#         super(CreateAccScreen, self).__init__()
#         loadUi("createacc.ui",self)
#         self.passwordfield.setEchoMode(QtWidgets.QLineEdit.Password)
#         self.confirmpasswordfield.setEchoMode(QtWidgets.QLineEdit.Password)
#         self.signup.clicked.connect(self.signupfunction)

#     def signupfunction(self):
#         user = self.emailfield.text()
#         password = self.passwordfield.text()
#         confirmpassword = self.confirmpasswordfield.text()

#         if len(user)==0 or len(password)==0 or len(confirmpassword)==0:
#             self.error.setText("Please fill in all inputs.")

#         elif password!=confirmpassword:
#             self.error.setText("Passwords do not match.")
#         else:
#             conn = sqlite3.connect("shop_data.db")
#             cur = conn.cursor()

#             user_info = [user, password]
#             cur.execute('INSERT INTO login_info (username, password) VALUES (?,?)', user_info)

#             conn.commit()
#             conn.close()

#             fillprofile = FillProfileScreen()
#             widget.addWidget(fillprofile)
#             widget.setCurrentIndex(widget.currentIndex()+1)

# class FillProfileScreen(QDialog):
#     def __init__(self):
#         super(FillProfileScreen, self).__init__()
#         loadUi("fillprofile.ui",self)
#         self.image.setPixmap(QPixmap('placeholder.png'))



# main
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(w)
    widget.setFixedHeight(800)
    widget.setFixedWidth(1200)
    widget.show()
    sys.exit(app.exec_())
