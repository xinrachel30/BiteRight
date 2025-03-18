import praw
import time

f = open("demofile2.txt", "a")
f.write("Now the file has more content!")
f.close()

reddit_read_only = praw.Reddit(client_id="CYzoKYjkZSnH9r5vRUz8fw", #Jordan's client ID
                    client_secret="Reajfk_GmsByCj_GHsDR-hsKmQ96LQ", # Jordan's client secret
                    user_agent="Biteright") #Jordan's user agent

thread_urls = []
thread_urls.append("https://www.reddit.com/r/EatCheapAndHealthy/comments/oo8uyu/what_are_some_stupidly_easy_food_pairings_that_go/")
thread_urls.append("https://www.reddit.com/r/Cooking/comments/oodevv/what_weird_food_combinations_have_you_been_served/?chainedPosts=t3_oo8uyu")

#Get all comments regardless of nested structure of comment replies
def getComments(ledger : dict[str,int], comments) -> dict[str,int]: 
  ''' Processes all comments regardless of nested structure of comment replies. 
      arg comments: Praw CommentForest
      Returns: dict[str,int] where the string is guarenteed to be lowercase
  '''
  for comm in comments: 
    ledger.update({str(comm.body).lower(): int(comm.score)})
    comm.replies.replace_more(limit=None)
    if comm.replies.list() != []: 
      ledger = ledger | getComments(ledger, comm.replies)
  return ledger

comm_score_dict = {}

for thisThread in thread_urls: 
  print("Loading Thread:\n")
  threadContent = reddit_read_only.submission(url=thisThread)
  while True:
    try:
        threadContent.comments.replace_more(limit=None)
        comm_score_dict = comm_score_dict | getComments({}, threadContent.comments)
        break
    except Exception:
        print("Waiting on replace_more...")
        time.sleep(5)

with open('tempDataset.txt', 'w', encoding='utf-8', errors='ignore') as f:
    f.write(str(comm_score_dict))