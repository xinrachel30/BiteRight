import praw
import time
import pickle
import os

reddit_read_only = praw.Reddit(client_id="CYzoKYjkZSnH9r5vRUz8fw", #Jordan's client ID
                    client_secret="Reajfk_GmsByCj_GHsDR-hsKmQ96LQ", # Jordan's client secret
                    user_agent="Biteright") #Jordan's user agent

thread_urls = []
thread_urls.append("https://www.reddit.com/r/EatCheapAndHealthy/comments/oo8uyu/what_are_some_stupidly_easy_food_pairings_that_go/") # (1.9K comments)
thread_urls.append("https://www.reddit.com/r/Cooking/comments/oodevv/what_weird_food_combinations_have_you_been_served/?chainedPosts=t3_oo8uyu") # (987 comments)
thread_urls.append("https://www.reddit.com/r/AskReddit/comments/jgoc5b/what_food_pairing_sounds_absurd_but_tastes_fine/") # (1.6K comments)
# following links added 4/17
thread_urls.append("https://www.reddit.com/r/AskReddit/comments/30aegf/what_two_weird_food_combinations_really_taste/") # (1.1K comments)
thread_urls.append("https://www.reddit.com/r/Cooking/comments/15gpnyl/best_food_combos/") # (33 comments)
thread_urls.append("https://www.reddit.com/r/AskReddit/comments/vza7nr/what_food_combos_are_surprisingly_good/?chainedPosts=t3_15gpnyl") # (2.5K comments)
thread_urls.append("https://www.reddit.com/r/foodhacks/comments/16njbvm/what_are_your_weird_bingefood_combos/?chainedPosts=t3_vza7nr%2Ct3_15gpnyl") # (1.1K comments)
thread_urls.append("https://www.reddit.com/r/nutrition/comments/mkgpf8/is_there_a_list_of_foods_that_go_together/") # (145 comments)
thread_urls.append("https://www.reddit.com/r/Cooking/comments/1cxeom1/what_is_a_simple_food_combo_you_never_thought_to/") # (2.2K comments)


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

with open(os.path.join("data", 'comment_score_dict.txt'), 'w', encoding='utf-8', errors='ignore') as f:
    f.write(str(comm_score_dict))

with open(os.path.join("data" ,"comment_score_dict.pkl"), "wb") as file:
    pickle.dump(comm_score_dict, file)