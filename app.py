import streamlit as st
import pandas as pd
import cv2
import pyttsx3
from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

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

	menu = ["Home","Login","SignUp","Text To Speech"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")

	elif choice == "Login":
		st.subheader("Login Section")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password", type='password')
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


if __name__ == '__main__':
	main()