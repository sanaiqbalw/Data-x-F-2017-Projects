from crawler.crawler import crawler
import re;
from bs4 import BeautifulSoup;
import math;
from datetime import datetime;
import json;

class ethereum(crawler) :
    @classmethod
    def __init__(self) :
        super(ethereum,self).__init__();

        self.__addressformat = "https://forum.ethereum.org/discussions/p{0}"
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
    def __loadHtml(self, page) :
        address = self.__getAddressFormat().format(page);

        response = super(ethereum,self).getResponse(address);

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

            f = open("ethereum"+"_"+str(pageno)+".json","wb");
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

        pages={};

        pages["posts"] = [];

        for currentpage in range(startpage, endpage+1) :
            result = self.crawlingPage(currentpage);
            pages["posts"] += result["posts"];

        return pages;

    @classmethod
    def __parsePostsInfo(self, pageno) :
        pageinfolist = [];

        discussionlist = self.__soupFindAll("tr", {"id":re.compile(r"Discussion_[0-9]+")});

        for discussion in discussionlist :
            titleobj = discussion.find("a",{"class":"Title"});

            pageinfo = {};
            pageinfo["uri"] = titleobj["href"];

            category = discussion.find(class_="MItem Category").a.string;
            if(re.search('MOVED:', titleobj.text) or \
                category == "Spanish" or \
                category == "Russian" or \
                category == "Romanian" or \
                category == "Portugese" or \
                category == "Japanese" or \
                category == "Hebrew" or \
                category == "French" or \
                category == "Italian" or \
                category == "German" or \
                category == "Chinese" or \
                category == "Turkish") :
                continue;

            commentsobj = discussion.find("td",{"class":"BigCount CountComments"});

            if(commentsobj == None) :
                continue;

            replyobj = commentsobj.find("span",{"class":"Number"});
            match = re.search('([0-9,]+)', replyobj['title']);
            pageinfo["reply"] = int(match.group(1).replace(',',''));

            countviewsobj  = discussion.find("td",{"class":"BigCount CountViews"});

            if(countviewsobj == None) :
                continue;

            viewobj = countviewsobj.find("span",{"class":"Number"});
            match = re.search('([0-9,]+)', viewobj['title']);
            pageinfo["views"] = int(match.group(1).replace(',',''));

            pageinfolist.append(pageinfo);

        return pageinfolist;

    @classmethod
    def __parsePost(self, uri, replycount) :
        response = self.getResponse(uri);
        soup = BeautifulSoup(response.text, "html.parser");
        
        posts = {};

        topic = soup.find("h1").text;
        discussionobj = soup.find("div",{"id":re.compile(r"Discussion")});

        time = discussionobj.find("time");
        # May 10, 2015  9:40PM
        dateobj = datetime.strptime(time["title"], "%B %d, %Y  %H:%M%p");
        datestr = dateobj.strftime("%Y-%m-%d %H:%M:%S");

        messageobj = discussionobj.find("div",{"class":"Message"});
        blockquotelist = messageobj.find_all("blockquote");
        for quote in blockquotelist :
            quote.decompose();

        posts["topic"] = topic;
        posts["date"] = datestr;
        posts["content"]=self.__removeTag(messageobj.prettify().split("\n"));

        commentlist = soup.find_all("li",attrs={"id":re.compile(r"Comment_[0-9]+")});

        replies = [];

        replies = self.__parseReply(soup);

        # for comment in commentlist :
        #     reply = {};
        #     commenttime = comment.find("time");
        #     commentdateobj = datetime.strptime(commenttime["title"], "%B %d, %Y  %H:%M%p");
        #     commentdatestr = dateobj.strftime("%Y-%m-%d %H:%M:%S");
        #     commentmessageobj = comment.find("div", {"class":"Message"});

        #     reply["date"] = commentdatestr;
        #     reply["content"] = commentmessageobj.text;

        #     replies.append(reply);

        if replycount > 30 :
            replypageno = int(replycount/30)+1;

            for currentreplypage in range(1, replypageno) :
                postresponse = self.getResponse(uri+"/p"+str(currentreplypage));
                postsoup = BeautifulSoup(postresponse.text, "html5lib");
                replies += self.__parseReply(postsoup);

        posts["replies"] = replies;

        return posts;

    @classmethod
    def __parseReply(self, soup) :
        replies = [];

        commentlist = soup.find_all("li",attrs={"id":re.compile(r"Comment_[0-9]+")});

        replies = [];

        for comment in commentlist :
            reply = {};
            time = comment.find("time");
            dateobj = datetime.strptime(time["title"], "%B %d, %Y  %H:%M%p");
            datestr = dateobj.strftime("%Y-%m-%d %H:%M:%S");

            messageobj = comment.find("div", {"class":"Message"});
            blockquotelist = messageobj.find_all("blockquote");
            for quote in blockquotelist :
                quote.decompose();

            reply["date"] = datestr;
            reply["content"] = self.__removeTag(messageobj.prettify().split("\n"));

            replies.append(reply);

        return replies;

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