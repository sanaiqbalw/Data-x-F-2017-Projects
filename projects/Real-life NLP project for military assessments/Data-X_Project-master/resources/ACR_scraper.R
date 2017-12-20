library(XML)
library(stringr)

df = list(Acronym = c(), Definition = c())

for (let in letters){
  print(let)
  #get the links for all the acronyms starting with letter let
  URL = str_c('http://www.dtic.mil/doctrine/dod_dictionary/acronym/', let, '/')
  search_results = htmlParse(readLines(URL))
  links = getHTMLLinks(search_results)
  acs = str_extract(links, "\\'[0-9]+\\.")
  acs = unlist(lapply(acs, substr, 2, 6))
  
  #create new URLs to query to find the definitions of each acronym
  URL2 = str_c(URL, acs, '.html')
  
  #loop through all the acronyms starting with current letter
  for (linkURL in URL2){
    Defhtml = readLines(linkURL)
    ACR = str_extract(Defhtml[4], "<title>[^<]+")
    ACR = substr(ACR, 8, nchar(ACR))
    Def = str_extract(Defhtml[8], "([\\t] |R/>)[^<]+")
    Def = substr(Def, 4, nchar(Def))
    df$Acronym = append(df$Acronym, ACR)
    df$Definition = append(df$Definition, Def)
  }
}

#do the acronyms starting with a number
URL = 'http://www.dtic.mil/doctrine/dod_dictionary/acronym/num/'
search_results = htmlParse(readLines(URL))
links = getHTMLLinks(search_results)
acs = str_extract(links, "\\'[0-9]+\\.")
acs = unlist(lapply(acs, substr, 2, 6))

#create new URLs to query to find the definitions of each acronym
URL2 = str_c(URL, acs, '.html')

#loop through all the acronyms starting with current letter
for (linkURL in URL2){
  Defhtml = readLines(linkURL)
  ACR = str_extract(Defhtml[4], "<title>[^<]+")
  ACR = substr(ACR, 8, nchar(ACR))
  Def = str_extract(Defhtml[8], "([\\t] |R/>)[^<]+")
  Def = substr(Def, 4, nchar(Def))
  df$Acronym = append(df$Acronym, ACR)
  df$Definition = append(df$Definition, Def)
}

df2 = data.frame(df)
write.csv(df2, file = 'ACRscrape.csv')
