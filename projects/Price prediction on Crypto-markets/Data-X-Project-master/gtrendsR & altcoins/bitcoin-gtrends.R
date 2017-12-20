setwd("C:\\Users\\SZXS4T\\Documents\\DX\\Sai-Projects\\Crypto-Prediction")
google_data <- gtrendsR::gtrends("bitcoin", geo = c("US"), time = "all")
datax = read.csv("coindesk-bpi-USD-ohlc_data-2010-2017.csv", header = T)
head(datax)
head(google_data)
write.csv(google_data, file = "MyData.csv")
x = datax$Date
head(x)
y = datax$Close
plot(x, y, type="l")



