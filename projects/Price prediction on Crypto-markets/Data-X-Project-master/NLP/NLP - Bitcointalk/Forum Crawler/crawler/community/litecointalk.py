# http://www.todayhumor.co.kr/board/view.php?table={tablename}&no={2}&s_no={2}

from crawler.crawler import crawler
from bs4 import BeautifulSoup;
import re;
from datetime import datetime;
import json;

class litecointalk(crawler) :
    @classmethod
    def __init__(self) :
        super(litecointalk,self).__init__();

        self.__addressformat = "https://litecointalk.org/index.php?board=1.{0}"
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

        response = super(litecointalk,self).getResponse(address);

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
            print("pageno : " + pageno);

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

            f = open("litecointalk"+"_"+str(pageno)+".json","wb");
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

            if(re.search('MOVED:', span.text)) :
                continue;

            tr = span.find_parent("tr");

            statsobj = tr.find("td",{"class":re.compile(r"stats")});

            replycount = int(re.search("([0-9]+) Replies",statsobj.text).group(1));

            views = int(re.search("([0-9]+) Views",statsobj.text).group(1));

            result = {"uri":uri,"views":views,"reply":replycount};

            postinfolist.append(result);

        return postinfolist;

    @classmethod
    def __parsePost(self, address, replycount) :

        post={};
        idx = 0;
        replies=[];

        postresponse = self.getResponse(address);
        soup = BeautifulSoup(postresponse.text, "html5lib");
        quickModForm = soup.find("form",{"id":"quickModForm"});

        postarealist = quickModForm.find_all("div", attrs={"class":re.compile(r"postarea")});

        for postarea in postarealist :
            if idx == 0 :
                post["topic"] = postarea.find("h5").text;
                smalltext = postarea.find("div", {"class":"smalltext"});
                post["date"] = self.__parseDate(smalltext);
                messageobj = postarea.find("div",{"id":re.compile(r"msg_[0-9]+"),"class":"inner"});
                messageobj = self.__removeQuote(messageobj);
                post["content"] = self.__removeTag(messageobj.prettify().split("\n"));
            else :
                reply = {};
                smalltext = postarea.find("div", {"class":"smalltext"});
                reply["date"] = self.__parseDate(smalltext);
                messageobj = postarea.find("div",{"id":re.compile(r"msg_[0-9]+"),"class":"inner"});
                messageobj = self.__removeQuote(messageobj);
                reply["content"] = self.__removeTag(messageobj.prettify().split("\n"));

                replies.append(reply);
            idx += 1;

        if(replycount >= 15) :
            replypageno = int(replycount/15)+1;

            for currentreplypage in range(1,replypageno) :
                result=self.__parseReply(address+str(currentreplypage*15));

                replies = replies + result;

        post["replies"] = replies;

        return post;

    @classmethod
    def __parseDate(self, dateobj) :
        if dateobj.strong != None :
            dateobj.strong.decompose();
        dateinfo = dateobj.text.replace("«","").replace("»","").strip(" ");
        datetimeobj = datetime.strptime(dateinfo, "%B %d, %Y, %H:%M:%S %p");
        return  datetimeobj.strftime("%Y-%m-%d %H:%M:%S");

    @classmethod
    def __removeQuote(self, messageobj) :
        for quoteheader in messageobj.find_all("div",{"class":"quoteheader"}) :
            quoteheader.decompose();
        for blockquote in messageobj.find_all("blockquote") :
            blockquote.decompose();
        for quotefooter in messageobj.find_all("div",{"class":"quotefooter"}) :
            quotefooter.decompose();

        return messageobj;

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

    @classmethod
    def __parseReply(self, address) :

        replies = [];

        postresponse = self.getResponse(address);
        soup = BeautifulSoup(postresponse.text, "html5lib");
        quickModForm = soup.find("form",{"id":"quickModForm"});

        postarealist = quickModForm.find_all("div", attrs={"class":re.compile(r"postarea")});

        for postarea in postarealist :
            reply = {};
            smalltext = postarea.find("div", {"class":"smalltext"});
            reply["date"] = self.__parseDate(smalltext);
            messageobj = postarea.find("div",{"id":re.compile(r"msg_[0-9]+"),"class":"inner"});
            messageobj = self.__removeQuote(messageobj);
            reply["content"] = self.__removeTag(messageobj.prettify().split("\n"));

            replies.append(reply);

        return replies;