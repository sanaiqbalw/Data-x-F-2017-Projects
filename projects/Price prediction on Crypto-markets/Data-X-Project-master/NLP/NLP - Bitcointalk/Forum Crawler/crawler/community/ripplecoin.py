import re;
from crawler.crawler import crawler
from bs4 import BeautifulSoup;
from datetime import datetime;
import json;

class ripplecoin(crawler) :

    @classmethod
    def __init__(self) :
        super(ripplecoin,self).__init__();

        self.__addressformat = "http://www.xrpchat.com/forum/5-general-discussion/?page={0}"
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

        response = super(ripplecoin,self).getResponse(address);

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

            f = open("ripplecoint"+"_"+str(pageno)+".json","wb");
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
        items = self.__soupFindAll("ul", {"class":"ipsDataItem_stats"});

        postinfolist=[];

        for item in items :
            postinfo={};

            li = item.parent;

            mainobj = li.find("div",{"class":"ipsDataItem_main"});

            uriobj = mainobj.find("a");

            postinfo["uri"]=uriobj["href"];

            lilist = item.find_all("li");

            # reply
            postinfo["reply"] = int(lilist[0].find("span",{"class":"ipsDataItem_stats_number"}).text.replace(',',''));
            # views
            postinfo["views"] = int(lilist[1].find("span",{"class":"ipsDataItem_stats_number"}).text.replace(',',''));

            postinfolist.append(postinfo);

        return postinfolist;

    @classmethod
    def __parsePost(self, uri, replycount) :

        idx = 0;

        postresponse = self.getResponse(uri);

        soup = BeautifulSoup(postresponse.text, "html5lib");
        post = {};

        replies = [];

        post["topic"] = soup.find("h1",{"class":"ipsType_pageTitle"}).text;

        articles = soup.find_all("article");

        for article in articles:
            time = article.find("time");
            dateobj = datetime.strptime(time["title"], "%m/%d/%Y  %I:%M  %p");
            datestr = dateobj.strftime("%Y-%m-%d %H:%M:%S");
            contentobj = article.find("div",{"data-role":"commentContent"});
            quotelist = contentobj.find_all("blockquote");

            for quote in quotelist :
                quote.decompose();

            if idx == 0 :
                post["content"]=self.__removeTag(contentobj.prettify().split("\n"));
                post["date"] = datestr;
            else :
                reply = {};

                reply["content"] = self.__removeTag(contentobj.prettify().split("\n"));
                reply["date"] = datestr;

                replies.append(reply);
            idx += 1;

        if replycount >= 15 :
            replypageno = int(replycount/15)+1;
            for currentreplypage in range(2,replypageno+1) :
                result=self.__parseReply(uri+"?page="+str(currentreplypage));

                replies = replies + result;

        post["replies"] = replies;

        return post;

    @classmethod
    def __parseReply(self, uri) :
        replies = [];
        postresponse = self.getResponse(uri);
        soup = BeautifulSoup(postresponse.text, "html5lib");

        articles = soup.find_all("article");
        for article in articles:
            time = article.find("time");
            dateobj = datetime.strptime(time["title"], "%m/%d/%Y %I:%M %p");
            datestr = dateobj.strftime("%Y-%m-%d %H:%M:%S");
            contentobj = article.find("div",{"data-role":"commentContent"});
            quotelist = contentobj.find_all("blockquote");
            for quote in quotelist :
                quote.decompose();

            reply = {};

            reply["content"] = self.__removeTag(contentobj.prettify().split("\n"));
            reply["date"] = datestr;

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