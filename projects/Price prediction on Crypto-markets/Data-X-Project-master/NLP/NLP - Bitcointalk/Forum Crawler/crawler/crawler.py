import requests;

class crawler(object) :

	

	@classmethod
	def __init__(self) :
		self.__debug = False;
		pass;

	@classmethod
	def getAddressFormat(self) :
		return "";

	@classmethod
	def getResponse(self, address) :

		headers={"Header":"Mozilla/5.0 (Windows NT 6.3; WOW64; rv:47.0) Gecko/20100101 Firefox/47.0"}

		response = requests.get(address, headers=headers);

		return response;

	@classmethod
	def debuglog(self, message) :
		if (self.__debug) :
			print(message);