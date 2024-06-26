import asyncio
import logging

from api.elevenlabs_api import *

class TTSManager:
    def __init__(self):
        logging.info("Elevenlabs TTSManager initialized.")
        self.current_process = None
        self.current_task = None
        self.queue = asyncio.Queue()  # Queue to manage speech requests
        self.processing_complete = asyncio.Event()
        self.processing_complete.set() # start wth no processing


    async def speak(self, text):
        if text == "<no response>" or text == "":
            return
        await self.queue.put(text)  # Add speech request to the queue
        if self.current_task is None and self.current_process is None:
            asyncio.create_task(self.process_queue())
        else:
            await self.cancel_current_speech()


    async def process_queue(self):
        
        while not self.queue.empty():
            self.processing_complete.clear()
            try:
                text = await self.queue.get()
                self.current_process, self.current_task = await stream_with_process_control(generate_audio_stream(text))

                # Wait for the streaming task to complete
                if self.current_task:
                    await self.current_task
                # Additionally, wait for the mpv process to exit
                if self.current_process:
                    await self.current_process.wait()  # Ensure process finishes before proceeding
            
            except asyncio.CancelledError:
                logging.info("TTSManager:process_queue:CancelledError")
            except BrokenPipeError:
                logging.info("TTSManager:process_queue:BrokenPipeError")
            except Exception as e:
                logging.info(f"TTSManager:process_queue: Exception during speech generation: {e}")
    
        self.processing_complete.set()
        self.current_task = None
        self.current_process = None


    async def cancel_current_speech(self):
        if self.current_process and self.current_task:
            await cancel_streaming(self.current_process, self.current_task)
            self.current_process = None
            self.current_task = None


    # wait until all the speech is done
    async def done(self):
        await self.processing_complete.wait()


    async def __aenter__(self):
        return self


    async def __aexit__(self, exc_type, exc, tb):
        while not self.queue.empty():
            await self.queue.get()  # Clear the queue
        await self.cancel_current_speech()

class DummyTTSManager(TTSManager):
    async def speak(self, text):
        print("DummyTTSManager:speak:", text)

    async def done(self):
        pass

    async def cancel_current_speech(self):
        pass


async def __test_async():
    async with TTSManager() as tts_manager:
        await tts_manager.speak("Not cancelled, say everything, long statement here.")
        await asyncio.sleep(13) # wait to finish

        print(">>> First")
        await tts_manager.speak("First message, it is cancelled and never says this part.")
        await asyncio.sleep(1)  # Simulate waiting for a while
        print(">>> Second interrupt")
        # This call will cancel the previous speech and start the new one
        await tts_manager.speak("This is the second message it is not cancelled.")

        await tts_manager.done()

if __name__ == "__main__":
    asyncio.run(__test_async())