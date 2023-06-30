import streamlit as st
import pandas as pd
import cv2
import pyttsx3
from PIL import Image
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

	st.title("Welcome to the home page")

	menu = ["Home","Login","SignUp","Text To Speech"]
	choice = st.sidebar.selectbox("Menu",menu)
	img = Image.open('sign 1.png')
st.image(img)
if choice == "Home":
         st.subheader("Home")

elif choice == "Login":
         st.subheader("Login Section")
         username = st.sidebar.text_input("User Name")
         password = st.sidebar.text_input("Password", type='password')
         if st.sidebar.checkbox("Login"):
                  # if password == '12345':
                  create_usertable()
                  hashed_pswd = make_hashes(password)
                  result = login_user(username, check_hashes(password, hashed_pswd))
                  if result:
                           st.title("Webcam Live Feed")
                           run = st.checkbox('Start Camera')
                           FRAME_WINDOW = st.image([])
                           camera = cv2.VideoCapture(0)
                  while run:
                           _, frame = camera.read()
                           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                           FRAME_WINDOW.image(frame)
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



