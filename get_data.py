"""
get_data.py
"""
import numpy as np
import urllib
import cv2
import os

import re
import requests
import urllib.request
from bs4 import BeautifulSoup

import praw
import string
import random


# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	# resp = urllib.urlopen(url)
	req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
	with urllib.request.urlopen(req) as u:
	    resp = u.read()
	image = np.asarray(bytearray(resp), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	# return the image
	return image



def non_reddit(site):
	"""
	"""
	response = requests.get(site)

	soup = BeautifulSoup(response.text, 'html.parser')
	img_tags = soup.find_all('img')

	urls = [img['src'] for img in img_tags]


	for url in urls:
		filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
		try: filename.group(1)
		except:continue
		# url=url.replace('//', '/')
		# if 'http' not in url:
		# 	# sometimes an image source can be relative
		# 	# if it is provide the base url which also happens
		# 	# to be the site variable atm.
		# 	url = '{}{}'.format(site, url.split('.com/')[-1])
		url='http:'+url
		# try:
		img = url_to_image(url)
		cv2.imshow('booch', img)
		cv2.waitKey(1000)

		cv2.imwrite('data/'+os.path.split(filename.group(1))[-1], img)




def main():
	with open('info.txt', 'rb') as f:
		lines=f.readlines()
		username=lines[0].decode().strip().replace("_", "")
		password=lines[1].decode().strip()
		client_id=lines[2].decode().strip()
		client_secret=lines[3].decode().strip()

	# name=input("Reddit Username: ")
	reddit = praw.Reddit(user_agent='Windows10:fetchData:v1.0 (by /u/{})'.format(username), client_id=client_id, client_secret=client_secret, username=username, password=password)
	print(reddit.read_only)
	print(reddit.user.me())
	subreddit = reddit.subreddit('Kombucha')
	conversedict={}
	hot_python=subreddit.hot(limit=1000)
	keywords=['mold', 'spots', 'smell', 'film', 'stuff', 'grayish', 'growth', 'scoby', 'pellicle', 'dot', 'worried', 'normal', 'mould', 'odor', '?']
	i=0
	samp_string =string.ascii_lowercase + string.ascii_uppercase+'0123456789' # for generating random sample strings
	with open("output.txt", 'wb') as f:
		for submission in hot_python:
			if not(any(string in submission.title for string in keywords)):
				continue
			if not(any(string in submission.url for string in ['.jpg', '.png'])):
				continue

			filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', submission.url)
			try: filename.group(1)
			except:continue
			print('\t', submission.title)
			print('\t', submission.selftext.replace('\n',"").replace('\r',"").replace('&#x200B',""))
			print(submission.num_comments)
			print(f"img number {i}")
			submission.comment_sort = 'top'
			submission.comments.replace_more(limit=0)
			for comment in submission.comments.list():
				if comment.author!=submission.author:
					# print('\t\t', comment.body.replace("\n", "").replace("\r", ""), comment.ups, comment.downs)
					clean_comment=re.sub('[^A-Za-z0-9 .?\-]+', '', comment.body.replace("\n", "").replace("\r", ""))
					print('\t\t', clean_comment, comment.ups, comment.downs)
			print('*'*60)


			img = url_to_image(submission.url)
			img=cv2.resize(img, (500, 500))
			while True:
				cv2.imshow('booch', img)
				key = cv2.waitKey(10)
				if key == ord('q'):
					break
				elif key in [ord(item) for item in 'ujnmkiol,.;p'] :
					fname = 'data/normal/normal_'+"".join(random.choice(samp_string) for i in range(16))
					cv2.imwrite(fname+'.jpg', img)
					with open(fname+'.txt', 'w') as f:
						clean_title=re.sub('[^A-Za-z0-9 .?\-]+', '', submission.title.replace("\n", "").replace("\r", ""))
						f.write(f'Title: {clean_title}\n')
						for comment in submission.comments.list():
							if comment.author!=submission.author:
								clean_comment=re.sub('[^A-Za-z0-9 .?\-]+', '', comment.body.replace("\n", "").replace("\r", ""))
								f.write('body: {}\nups: {}\ndowns: {}\n'.format(clean_comment, comment.ups,comment.downs))
					print('NORMAL')
					break
				elif key in [ord(item) for item in 'asdcxzfv']:
					print('MOLD')
					fname = 'data/sick/sick_'+"".join(random.choice(samp_string) for i in range(16))
					cv2.imwrite(fname+'.jpg', img)
					with open(fname+'.txt', 'w') as f:
						clean_title=re.sub('[^A-Za-z0-9 .?\-]+', '', submission.title.replace("\n", "").replace("\r", ""))
						f.write(f'Title: {clean_title}\n')
						for comment in submission.comments.list():
							if comment.author!=submission.author:
								clean_comment=re.sub('[^A-Za-z0-9 .?\-]+', '', comment.body.replace("\n", "").replace("\r", ""))
								f.write('body: {}\nups: {}\ndowns: {}\n'.format(clean_comment, comment.ups,comment.downs))
					break
				elif key==127:
					break

			i+=1
			# normal==l
			# mold==a

			continue


			if not submission.stickied:
				print('Title: {}, ups: {}, downs: {}, Have we visited?: {}, subid: {}'.format(submission.title,submission.ups,submission.downs,submission.visited, submission.id))
				print('created:{}, distinguished:{}, edited:{},num_comments:{}, score:{}, upvote_ratio:{}'.format(submission.created_utc, submission.distinguished, submission.edited, submission.num_comments, submission.score, submission.upvote_ratio))
				f.write(35*"-".encode()+"\r\n".encode())
				f.write('subid: {}, Title: {}, author:{}, ups: {}, downs: {}, Have we visited?: {}'.format(submission.id, submission.title,submission.author, submission.ups,submission.downs,submission.visited).encode())
				# f.write("\n".encode())
				f.write(' created:{}, distinguished:{}, edited:{},num_comments:{}, score:{}, upvote_ratio:{}\r\n'.format(submission.created_utc, submission.distinguished, submission.edited, submission.num_comments, submission.score, submission.upvote_ratio).encode())
				f.write('OPbody:{}\r\n'.format(submission.selftext.replace('\n',"").replace('\r',"").replace('&#x200B',"")).encode())
				# f.write("\n".encode())
				# author, clicked, comments, created_utc, distinguished, edited, id, is_video,
				# print(dir(submission))

				# link_flair_css_class, link_flair_text, locked, num_comments, over_18, permalink, score,
				# selftext, stickied, subreddit, subreddit_id, title, upvote_ratio
				submission.comments.replace_more(limit=0)
				print(submission)
				for comment in submission.comments.list():
					# print(dir(comment))
					f.write('id:{}, comment author:{}, body:{}, ups:{}, downs:{}, parent:{}\r\n'.format(comment.id, comment.author, comment.body,  comment.ups, comment.downs, comment.parent()).encode())
					# f.write("\n".encode())
					if comment.id not in conversedict.keys():
						conversedict[comment.id]=[comment.body, {}]
						if comment.parent()!=submission.id:
							parent=str(comment.parent())
							conversedict[parent][1][comment.id]=[comment.ups, comment.body]


if __name__=="__main__":
	# site = 'http://kombuchahome.com/how-to-tell-if-kombucha-scoby-has-mold-or-not/'
	# non_reddit(site)
	main()
