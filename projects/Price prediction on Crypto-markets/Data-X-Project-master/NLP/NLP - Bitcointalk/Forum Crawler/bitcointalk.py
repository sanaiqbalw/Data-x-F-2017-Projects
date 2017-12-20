# http://www.todayhumor.co.kr/board/view.php?table={tablename}&no={2}&s_no={2}

from crawler.crawler import crawler
from bs4 import BeautifulSoup;
import re;
from datetime import datetime;
import json;

class bitcointalk(crawler) :
	@classmethod
	def __init__(self) :
		super(bitcointalk,self).__init__();

		self.__addressformat = "https://bitcointalk.org/index.php?board=1.{0}"
		self.__soup = None;


	@classmethod
	def __getAddressFormat(self) :
		return self.__addressformat;

	@classmethod
	def __soupFind(self, tag, attrs) :
		return self.__soup.find(tag, attrs);

	@classmethod
	def __soupFindAll(self, tag, attrs) :
		return self.__soup.find_all(tag, attrs=attrs);

	@classmethod
	def __loadHtml(self, index) :
		page = str(40*(index-1));
		address = self.__getAddressFormat().format(page);

		response = super(bitcointalk,self).getResponse(address);

		html = response.text;

		self.__soup = BeautifulSoup(html, "html5lib");

	@classmethod
	def getHtml(self) :
		if(self.__soup is None) :
			return "";
		else :
			return self.__soup.prettify();

	@classmethod
	def crawlingPage(self, pageno) :
		if(pageno < 1) :
			pageno = 1;

		try :
			print("pageno : " + str(pageno));

			self.__loadHtml(pageno);

			postinfolist = self.__parsePostsInfo(pageno);

			result={};
			postlist=[];

			for postinfo in postinfolist:

				post = {};
				
				post = self.__parsePost(postinfo["uri"], postinfo["reply"]);
				post["views"] = postinfo["views"];


				postlist.append(post);

			result["posts"] = postlist;

			f = open("bitcointalk"+"_"+str(pageno)+".json","wb");
			f.write(json.dumps(result, ensure_ascii=False).encode('utf-8'));
			f.close();

			return result;
			
		except Exception as e:
			raise e;

	@classmethod
	def crawlingPages(self, startpage, endpage) :

		if startpage <= 0 :
			startpage = 1;
			
		if endpage <= 0 :
			endpage = 1;

		pages = {};

		pages["posts"] = [];

		for page in range(startpage, endpage+1) :
			pageresult = self.crawlingPage(page);

			pages["posts"] += pageresult["posts"];


		return pages;

	@classmethod
	def __parsePostsInfo(self, page) :
		spans = self.__soupFindAll("span", {"id":re.compile(r"msg_[0-9]+")});

		postinfolist = [];

		for span in spans :

			uri = span.a["href"];
			td = span.parent;

			tdlist = td.parent.find_all("td",{"class":td["class"]});

			result = {"uri":uri,"views":int(tdlist[2].text),"reply":int(tdlist[1].text)};

			postinfolist.append(result);

		return postinfolist[4:50];

	@classmethod
	def __parsePost(self, address, replycount) :

		post={};
		idx = 0;
		replies=[];

		postresponse = self.getResponse(address);
		soup = BeautifulSoup(postresponse.text, "html5lib");
		quickModForm = soup.find("form",{"id":"quickModForm"});
		tr = quickModForm.find("tr");
		trlist=quickModForm.find_all("tr",attrs={"class":tr["class"]});

		for tr in trlist :
			headerandpost = tr.find("td",{"class":"td_headerandpost"});

			subject = headerandpost.find("div",{"id":re.compile(r"subject")});
			datestr = subject.parent.find("div",{"class":"smalltext"}).text;
			postobj = headerandpost.find("div",{"class":"post"});
			print("Subject: " + str(subject))

			quotelist = postobj.find_all("div",{"class":"quoteheader"});
			for quote in quotelist :
				quote.decompose();

			if(idx == 0) :
				post["topic"]=subject.text;
				post["content"]=self.__removeTag(postobj.prettify().split("\n"));
				post["date"]=self.__parseDate(datestr);
			else :
				reply={};
				reply["date"]=self.__parseDate(datestr);
				reply["content"]=self.__removeTag(postobj.prettify().split("\n"));
				replies.append(reply);
			print ("post/reply number: " + str(idx))
			idx += 1;

		if(replycount >= 20) :
			replypageno = int(replycount/20)+1;



			for currentreplypage in range(1,replypageno) :

				result=self.__parseReply(address+str(currentreplypage*20));

				replies = replies + result;

		post["replies"]=replies;

		return post;

	@classmethod
	def __parseReply(self, address) :

		replies = [];
		
		postresponse = self.getResponse(address);

		soup = BeautifulSoup(postresponse.text, "html5lib");
		try:
			quickModForm = soup.find("form",{"id":"quickModForm"});
			tr = quickModForm.find("tr");
			trlist=quickModForm.find_all("tr",attrs={"class":tr["class"]});

			count = 0
			for tr in trlist :
				headerandpost = tr.find("td",{"class":"td_headerandpost"});

				subject = headerandpost.find("div",{"id":re.compile(r"subject")});
				datestr = subject.parent.find("div",{"class":"smalltext"}).text;
				postobj = headerandpost.find("div",{"class":"post"});
				for quote in postobj.find_all("div",{"class":"quoteheader"}) :
					quote.decompose();

				reply={};
				reply["date"]=self.__parseDate(datestr);
				reply["content"]= self.__removeTag(postobj.prettify().split("\n"));
				replies.append(reply);
				count = count + 1
				print ("post/reply number2: " + str(count))

			return replies;
		except:
			print("Error parsing post/reply")
			return None


	@classmethod
	def __parseDate(self,datestr):

		date="";
		if "Today" in datestr:
			split = datestr.split("at");
			date = datetime.now().strftime("%Y-%m-%d");
			date += split[1];
		else :
			dateobj = datetime.strptime(datestr,"%B %d, %Y, %I:%M:%S %p");
			date = dateobj.strftime("%Y-%m-%d %H:%M:%S");

		return date;

	@classmethod
	def __removeTag(self, lines) :
		result = "";

		for line in lines :
			line = re.sub('<[^>]*>','',line);
			line = re.sub('</[^>]*>','',line);
			line = re.sub('[\n\t]','',line);
			line = re.sub('\\\\n','',line);
			line = line.strip();

			if (len(line) > 0) :
				result += line+'\n';

		return result;