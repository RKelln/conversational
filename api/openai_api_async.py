import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import time

# env variables
load_dotenv()
my_key = os.getenv('OPENAI_API_KEY')

# OpenAI API
client = AsyncOpenAI(api_key=my_key)
assistant_id = os.getenv('OPENAI_ASSISTANT_ID')

# inspired by this post:
# https://community.openai.com/t/cant-add-messages-to-thread-while-a-run-is-active/491669/3

async def create_assistant():
    assistant = await client.beta.assistants.retrieve(assistant_id)
    return assistant

async def create_thread():
    thread = await client.beta.threads.create()
    return thread


async def add_message_to_thread(thread_id, user_question):
    # Create a message inside the thread
    message = await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_question
    )
    return message


async def get_answer(assistant_id, thread):
    # run assistant
    run =  await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    # wait for the run to complete
    last_update = time.time()
    while True:
        runInfo = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if runInfo.completed_at:
            # elapsed = runInfo.completed_at - runInfo.created_at
            # elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            #print(f"Run completed")
            print("")
            break
        if time.time() - last_update > 0.5:
            print(".", end="", flush=True)
            last_update = time.time()
        time.sleep(0.1)
    
    # Get messages from the thread
    messages = await client.beta.threads.messages.list(thread.id)
    message_content = messages.data[0].content[0].text.value
    return message_content


async def interactive_test():
    # Colour to print
        class bcolors:
            HEADER = '\033[95m'
            OKBLUE = '\033[94m'
            OKCYAN = '\033[96m'
            OKGREEN = '\033[92m'
            WARNING = '\033[93m'
            FAIL = '\033[91m'
            ENDC = '\033[0m'
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'
    
        # Create assistant and thread before entering the loop
        assistant = await create_assistant()
        print("Created assistant with id:" , f"{bcolors.HEADER}{assistant.id}{bcolors.ENDC}")
        thread = await create_thread()
        print("Created thread with id:" , f"{bcolors.HEADER}{thread.id}{bcolors.ENDC}")
        while True:
            question = input("> ")
            if "exit" in question.lower() or "quit" in question.lower():
                break
            
            # Add message to thread
            await add_message_to_thread(thread.id, question)
            message_content = await get_answer(assistant.id, thread)
            print(f"FYI, your thread is: , {bcolors.HEADER}{thread.id}{bcolors.ENDC}")
            print(f"FYI, your assistant is: , {bcolors.HEADER}{assistant.id}{bcolors.ENDC}")
            print(message_content)
        print(f"{bcolors.OKGREEN}Thanks and happy to serve you{bcolors.ENDC}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="../.env")

    asyncio.run(interactive_test())