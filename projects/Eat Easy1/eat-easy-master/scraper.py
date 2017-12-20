#import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from random import randint
from time import sleep
from difflib import SequenceMatcher
import pandas as pd

class Scraper():
	"""
	Searches AllMenus.com for a restaurant's menu items and corresponding descriptions.
	"""
	def __init__(self):
		self.site = 'https://www.allmenus.com'
		self.wait_time = 10 #60
		self.restuarant_name = None
		self.restuarant_addr = None
		self.bus_id = None
		self.sleep_min = 0 #3
		self.sleep_max = 1 #7
		self.data = []

	def startBrowser(self):
		self.browser = webdriver.PhantomJS()
		self.browser.implicitly_wait(self.wait_time)

	def searchRestuarant(self):
		print(">>>>>>>>>>")
		print('Starting search for {0}, {1}...'.format(self.restuarant_name, self.restuarant_addr))
		self.browser.get(self.site)
		# unchecking online ordering box
		check_box = self.browser.find_element(By.CSS_SELECTOR, "input#online-ordering-checkbox.s-checkbox-input")
		check_box.click()
		# entering address
		search_field = self.browser.find_element(By.ID, 'get-address')
		search_field.clear()
		search_field.send_keys(self.restuarant_addr)
		print('Sleeping...')
		sleep(randint(self.sleep_min, self.sleep_max))
		search_field.send_keys(Keys.RETURN)
		print('Sleeping...')
		sleep(randint(self.sleep_min, self.sleep_max))

	def selectRestaurantFromResults(self):
		print('Selecting desired restuarant from search results...')
		lst_results = self.browser.find_elements(By.CSS_SELECTOR, 'h4.name a')
		restuarant_href = None
		for result in lst_results:
			if not isinstance(result, type(None)):
				possible_href = result.get_attribute('href')
				if possible_href and self.restuarant_name.lower().replace(' ', '-').replace("'", "-") in possible_href:
					restuarant_href = possible_href
					break
		if restuarant_href:
			print('Sleeping...')
			sleep(randint(self.sleep_min, self.sleep_max))
			self.browser.get(restuarant_href)
		return restuarant_href

	def similar(self, a, b):
		return SequenceMatcher(None, a, b).ratio()

	def scrapeDesiredInfo(self):
		print('Scraping desired info...')
		#menu_items = []
		resulting_menu_items = self.browser.find_elements(By.CLASS_NAME, 'item-title')
		#item_descriptions = []
		resulting_item_descriptions = self.browser.find_elements(By.CSS_SELECTOR, 'p.description')
		#prices = []
		resulting_prices = self.browser.find_elements(By.CLASS_NAME, 'item-price')
		for i in range(len(resulting_menu_items)):
			info_dict = {
				'id': self.bus_id,
				'rest_name': self.restuarant_name,
				'menu_item': resulting_menu_items[i].text,
				'description': resulting_item_descriptions[i].text,
				'price': resulting_prices[i].text
			}
			self.data.append(info_dict)

	def write_to_csv(self):
		print("Writing to csv...")
		scraper_data = pd.DataFrame(columns=['rest_name', 'menu_item', 'description', 'price'], dtype=str) ##id
		scraper_data = scraper_data.append(self.data, ignore_index=True)
		scraper_data.to_csv('scraper_data.csv')

	def closeBrowser(self):
		if self.browser:
			self.browser.quit()

# Read in CSV + minor cleaning
print("Reading csv...")
phoenix_addresses = pd.read_csv('Phoenix_Addresses_Id.csv')
phoenix_addresses.dropna(inplace=True)

# Run Scraper
print("Starting scraper...")
scraper = Scraper()
scraper.startBrowser()
for i in range (175, phoenix_addresses.shape[0]):
	print(i)
	try:
		scraper.restuarant_addr = phoenix_addresses['address'][i].split(',')[0] +', '+ phoenix_addresses['city'][i] +', '+ phoenix_addresses['state'][i]
		scraper.restuarant_name = phoenix_addresses['name'][i]
		scraper.bus_id = phoenix_addresses['id'][i]
		scraper.searchRestuarant()
		success = scraper.selectRestaurantFromResults()
		if success:
			scraper.scrapeDesiredInfo()
	except KeyError:
		pass
scraper.closeBrowser()
scraper.write_to_csv()
