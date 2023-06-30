import streamlit as st
import pyttsx3
from function import *
from keras.models import model_from_json
import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets
import streamlit as st
from PIL import Image
import sys
camera_video = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=3, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=3, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

colors = []
for i in range(0, 20):
	colors.append((245, 117, 16))
print(len(colors))

# Security
#passlib,hashlib,bcrypt,scrypt

import sqlite3
import hashlib


def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management


conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions

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
		Dialog.setWindowTitle(_translate("Dialog1", "Dialog1"))
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
																					   thickness=2,
																					   circle_radius=2),
										  connection_drawing_spec=mp_drawing.DrawingSpec(color=(70, 150, 20),
																						 thickness=2,
																						 circle_radius=2))

		return output_image, results

	def countFingers(self, image, results, draw=True, display=True):
		height, width, _ = image.shape
		output_image = image.copy()

		count = {'RIGHT': 0, 'LEFT': 0}

		# Store the indexes of the tips landmarks of each finger of a hand in a list.
		fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
							mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

		fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False,
							'RIGHT_RING': False,
							'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False,
							'LEFT_MIDDLE': False,
							'LEFT_RING': False, 'LEFT_PINKY': False}

		for hand_index, hand_info in enumerate(results.multi_handedness):
			hand_label = hand_info.classification[0].label

			hand_landmarks = results.multi_hand_landmarks[hand_index]  # ladnmark recheck from hand

			for tip_index in fingers_tips_ids:

				# Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
				finger_name = tip_index.name.split("_")[0]

				if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
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
		# print(fingers_statuses)
		if draw:

			xx = list(fingers_statuses.keys())
			yy = list(fingers_statuses.values())

			if (xx[0] == 'RIGHT_THUMB' and yy[0] == True and yy[1] == False and yy[2] == False and yy[
				3] == False and yy[4] == False):
				cv2.putText(output_image, " Gesture Recognition ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1,
							(0, 0, 255), 2)
				cv2.putText(output_image, "All the best", (width // 4 - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
							(0, 0, 255), 10, 10)
				return output_image, fingers_statuses, count

		if draw:
			x = list(fingers_statuses.keys())
			y = list(fingers_statuses.values())
			cc1 = list(count.values())
			if (cc1[0] == 3 and cc1[1] == 0 and x[2] == 'RIGHT_MIDDLE' and x[3] == 'RIGHT_RING' and x[
				4] == 'RIGHT_PINKY' and y[2] == True and y[3] == True and y[4] == True and y[0] == False and y[
				1] == False):
				cv2.putText(output_image, " Gesture Recognition ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1,
							(0, 0, 255), 2)
				cv2.putText(output_image, "Nice", (width // 4 - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
							(0, 0, 255), 10, 10)
				return output_image, fingers_statuses, count

		# print(fingers_statuses)
		if draw:

			xx = list(fingers_statuses.keys())
			yy = list(fingers_statuses.values())
			cc2 = list(count.values())
			if (cc2[0] >= 0 and cc2[1] == 0 and xx[1] == 'RIGHT_INDEX' and xx[2] == 'RIGHT_MIDDLE' and yy[
				0] == False and yy[1] == True and yy[2] == True and yy[3] == False and yy[4] == False):
				cv2.putText(output_image, " Gesture Recognition ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1,
							(0, 0, 255), 2)
				cv2.putText(output_image, "Victory", (width // 4 - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
							(0, 0, 255), 10, 10)
				return output_image, fingers_statuses, count

		r_count = list(count.values())

		if draw:
			cc1 = list(count.values())

			if (r_count[0] == 0 and hand_label == "Right"):

				cc = str(sum(count.values()))
				cv2.putText(output_image, " Gesture Recognition ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1,
							(0, 0, 255), 2)
				cv2.putText(output_image, "THANK YOU", (width // 2 - 150, 130), cv2.FONT_HERSHEY_SIMPLEX, 2,
							(0, 0, 255), 10, 10)
				return output_image, fingers_statuses, cc
			elif (cc1[0] == 5 and cc1[1] == 0 and r_count[0] == 5 and hand_label == "Right"):

				cc = str(sum(count.values()))
				cv2.putText(output_image, " Gesture Recognition ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1,
							(0, 0, 255), 2)
				cv2.putText(output_image, "HIGHFIVE", (width // 2 - 150, 130), cv2.FONT_HERSHEY_SIMPLEX, 2,
							(0, 0, 255), 10, 10)
				return output_image, fingers_statuses, cc

			cv2.putText(output_image, " Gesture Recognition ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1,
						(0, 0, 255), 2)
			cv2.putText(output_image, str(sum(count.values())), (width // 2 - 150, 140),
						cv2.FONT_HERSHEY_SIMPLEX,
						5, (0, 0, 255), 10, 10)

			cc = str(sum(count.values()))
		# time.sleep(2)
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
			# print(count)
			cv2.imshow('Gesture Recognition', frame)
			k = cv2.waitKey(1) & 0xFF
			if (k == 27):
				break
		camera_video.release()
		cv2.destroyAllWindows()


def prob_viz(res, actions, input_frame, colors, threshold):
	output_frame = input_frame.copy()
	for num, prob in enumerate(res):
		cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
		cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
					cv2.LINE_AA)

	return output_frame

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(name TEXT,email TEXT,mobile TEXT,username TEXT,password TEXT)')


def add_userdata(name,email,mobile,username,password):
	print(name)
	print(email)
	c.execute('INSERT INTO userstable(name,email,mobile,username,password) VALUES (?,?,?,?,?)',(name,email,mobile,username,password))
	conn.commit()


def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return  data


def main():
	"""Simple Login App"""
	

	st.title("Sign Language Recognition")

	menu = ["Home","Login","SignUp","Text To Speech","Gesture Recognition"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader(""" Discover the beauty of Sign Language and bridge the communication gap ! """)
		image = Image.open("pic3.png")
		st.image(image, caption='', use_column_width=True)
		

	elif choice == "Login":
		st.subheader("Login Section")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password", type='password')
		image = Image.open("pic1.png")
		st.image(image, caption='', use_column_width=True)
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username, check_hashes(password, hashed_pswd))
			if result:
				sequence = []
				sentence = []
				accuracy = []
				predictions = []
				threshold = 0.8
				st.subheader("Starting Camera Module")

				# st.title("Webcam Live Feed")
				# run = st.checkbox('Start Camera')
				FRAME_WINDOW = st.image([])
				cap = cv2.VideoCapture(0)
				with mp_hands.Hands(
						model_complexity=0,
						min_detection_confidence=0.5,
						min_tracking_confidence=0.5) as hands:

					while cap.isOpened():

						# Read feed
						ret, frame = cap.read()

						# Make detections
						cropframe = frame[40:400, 0:300]
						# print(frame.shape)
						frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
						# frame=cv2.putText(frame,"Active Region",(75,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,255,2)
						image, results = mediapipe_detection(cropframe, hands)
						# print(results)

						# Draw landmarks
						# draw_styled_landmarks(image, results)
						# 2. Prediction logic
						keypoints = extract_keypoints(results)
						sequence.append(keypoints)
						sequence = sequence[-30:]

						try:
							if len(sequence) == 30:
								res = model.predict(np.expand_dims(sequence, axis=0))[0]
								print(actions[np.argmax(res)])
								predictions.append(np.argmax(res))

								# 3. Viz logic
								if np.unique(predictions[-10:])[0] == np.argmax(res):
									if res[np.argmax(res)] > threshold:
										if len(sentence) > 0:
											if actions[np.argmax(res)] != sentence[-1]:
												sentence.append(actions[np.argmax(res)])
												accuracy.append(str(res[np.argmax(res)] * 100))
										else:
											sentence.append(actions[np.argmax(res)])
											accuracy.append(str(res[np.argmax(res)] * 100))

								if len(sentence) > 1:
									sentence = sentence[-1:]
									accuracy = accuracy[-1:]

						# Viz probabilities
						# frame = prob_viz(res, actions, frame, colors,threshold)
						except Exception as e:
							# print(e)
							pass

						cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
						# cv2.putText(frame,"Output: "+' '.join(sentence)+''.join(accuracy), (3,30),
						cv2.putText(frame, "Output: " + ''.join(sentence) + '', (3, 30),
									cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

						# Show to screen
						cv2.imshow('OpenCV Feed', frame)

						# Break gracefull
						k = cv2.waitKey(1) & 0xFF
						if (k == 27):
							break
					cap.release()
					cv2.destroyAllWindows()



			else:
				st.warning("Incorrect Username/Password")

	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_name = st.text_input("Enter Name")
		new_email = st.text_input("Email")
		new_mobile = st.text_input("Mobile No")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password", type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_name,new_email,new_mobile,new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")

	elif choice == "Text To Speech":
		st.subheader("Text To Speech Conversion")
		new_text = st.text_input("Enter Text")

		if st.button("Text to speech"):
			text_speech = pyttsx3.init()
			text_speech.setProperty("rate", 150)
			text_speech.say(new_text)
			text_speech.runAndWait()

	elif choice == "Gesture Recognition":

		st.subheader("Gesture Recognition")
		app = QtWidgets.QApplication(sys.argv)
		Dialog = QtWidgets.QDialog()
		ui = hand_detect()
		ui.setupUi(Dialog)
		Dialog.show()
		sys.exit(app.exec_())


if __name__ == '__main__':
	main()


