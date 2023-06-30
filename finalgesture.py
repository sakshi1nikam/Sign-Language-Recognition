import cv2


import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets


camera_video = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=3, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=3, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


class hand_detect:
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog1")
        Dialog.resize(887, 589)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(170, 260, 201, 34))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(450, 260, 201, 34))
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.pushButton.clicked.connect(self.start)

    def start(self):
        self.main()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog1","Dialog1"))
        self.pushButton.setText(_translate("Dialog1", "Start Gesture Recognition"))
        self.pushButton_2.setText(_translate("Dialog1", "Stop gesture recognition"))



    def detectHandsLandmarks(self, image, hands, draw=True, display=True):

        output_image = image.copy()
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
                                          connections=mp_hands.HAND_CONNECTIONS,

                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                       thickness=2, circle_radius=2),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(70, 150,20),
                                                                                      thickness=2, circle_radius=2))

        return output_image, results

    def countFingers(self, image, results, draw=True, display=True):
        height, width, _ = image.shape
        output_image = image.copy()


        count = {'RIGHT': 0, 'LEFT': 0}

        # Store the indexes of the tips landmarks of each finger of a hand in a list.
        fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

        fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                            'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                            'LEFT_RING': False, 'LEFT_PINKY': False}

        for hand_index, hand_info in enumerate(results.multi_handedness):
            hand_label = hand_info.classification[0].label


            hand_landmarks = results.multi_hand_landmarks[hand_index] #ladnmark recheck from hand

            for tip_index in fingers_tips_ids:

                # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
                finger_name = tip_index.name.split("_")[0]

                if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index-2].y):
                    fingers_statuses[hand_label.upper() + "_" + finger_name] = True

                    # Increment the count of the fingers up of the hand by 1.
                    count[hand_label.upper()] += 1

            # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
            thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x

            # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
            if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (
                    hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
                # Update the status of the thumb in the dictionary to true.
                fingers_statuses[hand_label.upper() + "_THUMB"] = True

                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1
        #print(fingers_statuses)
        if draw:

            xx = list(fingers_statuses.keys())
            yy = list(fingers_statuses.values())

            if (xx[0] == 'RIGHT_THUMB' and yy[0] == True and yy[1] == False and yy[2]== False and yy[3]== False and yy[4]== False):
                cv2.putText(output_image, " Gesture Recognition ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(output_image, "All the best", (width // 4 - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 10, 10)
                return output_image, fingers_statuses, count

        if draw:
            x = list(fingers_statuses.keys())
            y = list(fingers_statuses.values())
            cc1= list(count.values())
            if (cc1[0] == 3 and cc1[1] == 0 and x[2] == 'RIGHT_MIDDLE' and x[3] == 'RIGHT_RING' and x[4] == 'RIGHT_PINKY' and y[2] == True and y[3] == True and y[4] == True and y[0] == False and y[1] == False):
                cv2.putText(output_image, " Gesture Recognition ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(output_image, "Nice", (width // 4 - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 10, 10)
                return output_image, fingers_statuses, count

        #print(fingers_statuses)
        if draw:

            xx = list(fingers_statuses.keys())
            yy = list(fingers_statuses.values())
            cc2= list(count.values())
            if (cc2[0] >=0  and cc2[1] == 0 and xx[1] == 'RIGHT_INDEX'  and xx[2] == 'RIGHT_MIDDLE' and  yy[0] == False and yy[1] == True and yy[2]== True and yy[3]== False and yy[4]== False):
                cv2.putText(output_image, " Gesture Recognition ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(output_image, "Victory", (width // 4 - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 10, 10)
                return output_image, fingers_statuses, count



        r_count = list(count.values())

        if draw:
            cc1= list(count.values())

            if (r_count[0] == 0 and hand_label == "Right"):

                cc = str(sum(count.values()))
                cv2.putText(output_image, " Gesture Recognition " , (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(output_image, "THANK YOU", (width // 2 - 150, 130), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 10, 10)
                return output_image, fingers_statuses, cc
            elif (cc1[0] == 5 and cc1[1] == 0 and  r_count[0] == 5 and hand_label == "Right"):

                cc = str(sum(count.values()))
                cv2.putText(output_image, " Gesture Recognition ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(output_image, "HIGHFIVE", (width // 2 - 150, 130), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 10, 10)
                return output_image, fingers_statuses, cc

            cv2.putText(output_image, " Gesture Recognition ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.putText(output_image, str(sum(count.values())), (width // 2 - 150, 140), cv2.FONT_HERSHEY_SIMPLEX,
                        5, (0, 0, 255),10, 10)

            cc= str(sum(count.values()))
        #time.sleep(2)
        return output_image, fingers_statuses, cc
    def main(self):

        cv2.namedWindow('Fingers Counter', cv2.WINDOW_NORMAL)
        while camera_video.isOpened():
            ok, frame = camera_video.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame, results = self.detectHandsLandmarks(frame, hands_videos, display=False)

            # Check if the hands landmarks in the frame are detected.
            if results.multi_hand_landmarks:
                frame, fingers_statuses, count = self.countFingers(frame, results, display=False)
                #print(count)
            cv2.imshow('Gesture Recognition', frame)
            k = cv2.waitKey(1) & 0xFF
            if (k == 27):
                break
        camera_video.release()
        cv2.destroyAllWindows()




if (__name__ == "__main__"):

    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = hand_detect()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
