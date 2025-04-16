###DO NOT CONTINUOUSLY RUN THIS SCRIPT
###Minor costs and daily request limits are associated

from openai import OpenAI   #pip install openai
import asyncio              #pip install asyncio
import aiofiles             #pip install aiofiles 
import pickle
import time 
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "python_scripts_and_data", "data")

### If a thread executes in 1 second an 8 thread limit
### ensures we do not exceed 500 requests per minute
semaphore = asyncio.Semaphore(8)  #Limit concurrent thread execution to 8
write_lock = asyncio.Lock()       #Lock to control writing to file

async def useChat_and_write(chat, food): 
  async with semaphore: 
    print("Exexcuting " + str(food) + " thread:")
    myFormat = "Only reply with a comma separated list. Do this five times on separate lines. Each line must have at least ten adjectives."
    prompt = "Using online resources, explain to me what " + str(food) + " tastes like."
    response = chat.responses.create(
      model="gpt-4.1-nano",
      instructions=myFormat,
      input=prompt,
      max_output_tokens=200
    )

    async with write_lock: 
      async with aiofiles.open(os.path.join(DATA_DIR, "food_and_flavors.txt"), "a") as file: 
        await file.write("\n<" + str(food) + "/>\n")  #Separates responses from each other
        await file.write(response.output_text)
        #await file.write(response)
      print(str(food) + " results written to file!")


async def main(): 
  async with aiofiles.open(os.path.join(DATA_DIR, "food_and_flavors.txt"), "w") as file: 
    await file.write("")  #Creates file and clears file

  with open(os.path.join(DATA_DIR, "foodVocab.pkl"), "rb") as file:
    foods_list = pickle.load(file)
    time.sleep(1) #Increase likelihood that food_list loads on time
    print("foods_list is loaded")

  client = OpenAI()
  tasks = [useChat_and_write(client, food) for food in foods_list]
  results = await asyncio.gather(*tasks)

asyncio.run(main())

