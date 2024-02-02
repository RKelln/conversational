import os
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from dotenv import dotenv_values

assistant_id = "asst_FiGxIhtqI67jvN0TM48OCxCX"

class Conversation:
    client = None
    assistant = None
    thread = None
    run = None

    def __init__(self, client=None, assistant=None, thread=None, run=None):
        self.client = client
        self.assistant = assistant
        self.thread = thread
        self.run = run
    
    def status(self):
        if self.run is None:
            return None
        run = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=self.run.id)
        return run.status

    def messages(self):
        return self.client.beta.threads.messages.list(thread_id=self.thread.id)
    
    # def send_message(self, message):
    #     message = self.client.beta.threads.messages.create(
    #         thread_id=self.thread.id,
    #         role="user",
    #         content=message
    #     )
    #     return message
    
    def send_message(self, message):
        # Check if current run is active before sending a new message
        status = self.status()
        if status == 'in_progress':
            print("Waiting for the current run to complete before sending a new message...")
            return None
        elif status is None:
            # If run is not active, create message and start run
            message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=message
            )
            self.run = client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                #instructions="Please address the user as fire-tender.",
            )
            return message
        else:
            print("send_message: Run status: ", status)
            return None


    def end(self):
        if self.status() == "in_progress":
            self.client.beta.threads.runs.cancel(thread_id=self.thread.id, run_id=self.run.id)
        self.client.beta.threads.delete(self.thread.id)


def get_client():
  # Note: set the OPENAI_API_KEY environment variable
  return OpenAI(api_key=dotenv_values(".env")["OPENAI_API_KEY"])


def generate_images(client, prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1) -> str:
    
    # response is ImageReponse
    # https://github.com/openai/openai-python/blob/main/src/openai/types/images_response.py
    # response.data[0]
    # .b64_json: Optional[str] = None
    #       The base64-encoded JSON of the generated image, if `response_format` is `b64_json`.
    # .revised_prompt: Optional[str] = None
    #       The prompt that was used to generate the image, if there was any revision to the prompt.
    # .url: Optional[str] = None
    #       The URL of the generated image, if `response_format` is `url` (default).

    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        print(e)
        return None


def get_assistant(client):
    return client.beta.assistants.retrieve(assistant_id)

    # assistant = client.beta.assistants.create(
    #     name="Sohkepayin",
    #     instructions=assistant_instructions,
    #     model="gpt-4-turbo-preview"
    # )
    #return assistant


def start_conversation(client, assistant) -> Conversation:
    c = Conversation()
    c.client = client
    c.assistant = get_assistant(client)
    c.thread = client.beta.threads.create()
    c.run = None # don't start run until we send a message
    return c


def delete_assistant(client, assistant):
    client.beta.assistants.delete(assistant.id)

def stop_runs(assistant, client):
    runs = client.beta.threads.runs.list(assistant_id=assistant.id)

    print("existing runs: ", len(runs))
    for run in runs:
        print(run.status)
        run = client.beta.threads.runs.cancel(thread_id=run.thread.id, run_id=run.id)
        print(run.status)


if __name__ == "__main__":
    import time
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="../.env")

    test_messages = [
        "Hello?",
        "Uh, yeah, uh, I guess I'm just wondering, uh, what's the meaning of life?",
    ]

    try:
        client = get_client()
        assistant = get_assistant(client)
        print("checking assistant status. ")
        #stop_runs(assistant, client)
        
        conv = start_conversation(client, assistant)
        print("conversation started")
        conv.send_message(test_messages.pop(0))

        while len(test_messages) > 0:
            status = conv.status()

            if status in ["completed", "failed", "cancelled"]:
                # Handle end of conversation
                print(f"Conversation ended with status: {status}")
                if status == "completed":
                    messages = conv.messages()
                    print("messages: ")
                    for message in messages:
                        assert message.content[0].type == "text"
                        print({"role": message.role, "message": message.content[0].text.value})
                    conv.send_message(test_messages.pop(0))
                else:
                    break
            else:
                print("in progress...")
                time.sleep(1)

    except APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e)