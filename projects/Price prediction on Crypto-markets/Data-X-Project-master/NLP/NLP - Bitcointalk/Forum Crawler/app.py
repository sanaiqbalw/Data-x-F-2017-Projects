import sys;
import json;

# from crawler.community.bitcointalk import bitcointalk;
# from crawler.community.ripplecoin import ripplecoin;
# from crawler.community.ethereum import ethereum;
# from crawler.community.litecointalk import litecointalk;

from bitcointalk import bitcointalk

configpath = "";
if len(sys.argv) < 2 :
	configpath = "./default-config.json";
else :
	configpath = sys.argv[1];

crawler = None;

with open(configpath, "r") as configfile :
	config = json.loads(configfile.read());

	print("community : " + config["community"]);
	if (config["community"] == "bitcointalk") :
		crawler = bitcointalk();
	elif (config["community"] == "ripplecoin") :
		crawler = ripplecoin();
	elif (config["community"] == "ethereum") :
		crawler = ethereum();
	elif (config["community"] == "litecointalk") :
		crawler = litecointalk();
	else :
		print("There is no crawler");

	if(crawler == None) :
		sys.exit();

	if("pages" in config) :
		print("start crawling pages : " + str(config["pages"]["startpage"]) + " to : " + str(config["pages"]["endpage"]));
		crawler.crawlingPages(int(config["pages"]["startpage"]), int(config["pages"]["endpage"]));

	if("page" in config) :
		print("start crawling page : " + str(config["page"]["pageno"]));
		crawler.crawlingPage(int(config["page"]["pageno"]));

# crawler = bitcointalk();
# crawler.crawlingPage(1);
# crawler.crawlingPages(1,5);

# crawler = ripplecoin();
# crawler.crawlingPages(1,10);

# crawler = ethereum();
# crawler.crawlingPages(1,10);

# crawler = litecointalk();
# crawler.crawlingPages(1, 10);