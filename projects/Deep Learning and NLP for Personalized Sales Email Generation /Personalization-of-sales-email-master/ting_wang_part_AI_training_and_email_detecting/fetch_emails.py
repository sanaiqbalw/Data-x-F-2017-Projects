# -*- coding: utf-8 -*-

import sys
import smtplib
import json
import imaplib
import getpass
import email
import datetime
from reply_q import reply
import time

credentials = ('chatbot2017fall@gmail.com', 'xuansefeng')
email_sender = smtplib.SMTP('smtp.gmail.com', 587)
email_sender.ehlo()
email_sender.starttls()
email_sender.login(*credentials)
email_reader = imaplib.IMAP4_SSL('imap.gmail.com')
email_reader.login(*credentials)



def send_email(subject, body, to_address):
    body = 'Subject: {}\n\n{}'.format(subject, body)
    email_sender.sendmail('Automail_nlp@berkeley.edu', to_address, body)



def read_all_emails_generator(M):
  rv, data = M.search(None, '(UNSEEN)')
  #rv, data = M.search(None, '(ALL)')
  if rv != 'OK':
      print "No messages found!"
      return

  for num in data[0].split():
      rv, data = M.fetch(num, '(RFC822)')
      if rv != 'OK':
          print "ERROR getting message", num
          continue

      msg = email.message_from_string(data[0][1])
      subject = msg['Subject']
      print 'Message %s: %s' % (num, subject)
      print 'Raw Date:', msg['Date']
      date_tuple = email.utils.parsedate_tz(msg['Date'])
      if date_tuple:
          local_date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
          print "Local Date:", local_date.strftime("%a, %d %b %Y %H:%M:%S")
      body = []
      if msg.is_multipart():
          for payload in msg.get_payload():
              # if payload.is_multipart(): ...
              body.append(payload.get_payload())
              print 'Body is: \n', body[-1]
      else:
          body.append(msg.get_payload())
          print 'Body is: \n', body[-1]
      sender_name, sender_address = email.utils.parseaddr(msg['From'])
      yield {'body': '\n'.join(body), 'subject': subject, 'local_date': local_date.strftime("%a, %d %b %Y %H:%M:%S"), 'sender_address': sender_address, 'sender_name': sender_name}


def answer_questions(msg):
    lines = msg['body'].splitlines()
    resp = []
    for subline in lines:
        if subline.strip():
            resp.append(reply(subline, msg['sender_name']))
        # sublines = subline.split('.|?|!')
        # for line in sublines:
        #     if line.strip():
        #         resp.append(reply(line, msg['sender_name']))
    ans =''.join(resp)
    ans = '\n'.join([s[0].upper() + s[1:] for s in ans.splitlines()])
    return ans


def get_greeting(time):
    #"Local Date:", local_date.strftime("%a, %d %b %Y %H:%M:%S") Fri, 01 Dec 2017 21:33:11
    global greeting
    hour = str(time).split(':',1)[0][-2:]
    # if int(hour) < 10:
    #     greeting = 'Good morning'
    if int(hour) < 18:
        greeting = 'Good morning'
    else:
        greeting = 'Good evening'
    return greeting

greeting_end = 'Cheers,\nRobot'

def main():
    rv, mailboxes = email_reader.list()
    print mailboxes
    rv, data = email_reader.select('INBOX')
    if rv == 'OK':
        print "Processing mailbox...\n"
        for msg in read_all_emails_generator(email_reader):
            resp = '{}, {}, \n\n {},\n\n{}'.format(get_greeting(msg['local_date']), msg['sender_name'], answer_questions(msg),greeting_end)
            send_email('Re: ' + msg['subject'], resp, msg['sender_address'])
            print('email sent')
        #email_reader.close()
    #email_reader.logout()


main()

while True:
    main()
    time.sleep(5.0)
