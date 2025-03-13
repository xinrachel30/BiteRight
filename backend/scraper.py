from bs4 import BeautifulSoup as BS
import requests

with open("page1.html", "r", encoding="utf-8") as f: 
  doc = BS(f, "html.parser")

tags = doc.find_all("p")
#print(tags)
#print(type(tags))

myData = []
startDelim = "<p>"
endDelim = "</p>"

parseString = str(tags)
while parseString != "": 
  keyStr = ""
  junkStr, _, parseString = parseString.partition(startDelim)
  keyStr, _, parseString = parseString.partition(endDelim)
  if(keyStr == ""): 
    break
  myData.append(keyStr)

tags = doc.find_all("shreddit-comment-action-row")
#print(tags)

myScores = []

parseString = str(tags)
startDelim = "score=\""
endDelim = "\""
while parseString != "": 
  keyStr = ""
  junkStr, _, parseString = parseString.partition(startDelim)
  keyStr, _, parseString = parseString.partition(endDelim)
  if(keyStr == ""): 
    break
  myScores.append(keyStr)

myScores = myScores[0:len(myData)]

my_dict_idx = {k: v for k, v in zip(enumerate(myScores), myScores)}
my_dict = {k: v for k, v in zip(myData, myScores)}

myIdx = 1
for k , v in my_dict.items(): 
  print(str(myIdx) + ". " + str(k) + "\t" + "Score: " + str(v) + "\n")
  myIdx += 1

