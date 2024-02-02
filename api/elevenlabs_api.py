import asyncio
import logging
import os
import time

from elevenlabs import generate

# env variables
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('ELEVENLABS_API_KEY')


async def stream_audio_chunks(mpv_process, audio_stream):
    writer = mpv_process.stdin
    try:
        for chunk in audio_stream:
            if chunk is not None:
                writer.write(chunk)  # asyncio subprocess stdin is a StreamWriter
                await writer.drain()  # Ensure data is sent
    except BrokenPipeError:
        logging.info("stream_audio_chunks: BrokenPipeError caught")
    except Exception as e:
        logging.info(f"stream_audio_chunks: Error streaming audio chunk: {e}")
    finally:
        writer.close()
        await writer.wait_closed()  # Properly close the writer
        

async def create_mpv_process():
    # Use asyncio subprocess to create the mpv process
    mpv_command = ["mpv", "--no-cache", "--no-terminal", "--idle=no", "--loop=no", "--", "fd://0"]
    return await asyncio.create_subprocess_exec(
        *mpv_command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def stream_with_process_control(audio_stream):
    mpv_process = await create_mpv_process()
    # Start streaming audio chunks in a background task
    streaming_task = asyncio.create_task(stream_audio_chunks(mpv_process, audio_stream))
    return mpv_process, streaming_task


async def cancel_streaming(mpv_process, streaming_task):
    if mpv_process:
        mpv_process.terminate()  # Send termination signal to the subprocess
        await mpv_process.wait()
    if streaming_task:
        try:
            # Cancel the streaming task
            streaming_task.cancel()
            await streaming_task  # Await the task to handle cancellation
        except asyncio.CancelledError:
            logging.info("Streaming task was cancelled successfully.")
        except Exception as e:
            logging.info(f"Exception during streaming task cancellation: {e}")


async def stream_audio_async(audio_stream):
    mpv_process, streaming_task = await stream_with_process_control(audio_stream)
    return mpv_process, streaming_task


async def __test_async_cancel():
    print("Test async cancel")
    # Simulate generating an audio stream
    audio_stream = generate_audio_stream("This is an asynchronous streaming voice that runs for longer than 1 second so this part should be cut off!", test=True)
    # Start streaming and get the process and task
    mpv_process, streaming_task = await stream_audio_async(audio_stream)
    
    print("Sleeping")
    await asyncio.sleep(2)
    print("Wait done")

    # Now, let's cancel the streaming
    await cancel_streaming(mpv_process, streaming_task)

    print("Test async done")


async def __test_async():
    print("Test async")
    # Simulate generating an audio stream
    audio_stream = generate_audio_stream("This is an asynchronous streaming voice!", test=True)
    # Start streaming and get the process and task
    mpv_process, streaming_task = await stream_audio_async(audio_stream)
    print("await task")
    await streaming_task  # Wait for the streaming task to complete
    print("Task done")
    await mpv_process.wait()  # Ensure process finishes before proceeding
    print("Process done")
    print("Test async done")


def generate_audio_stream(text, test=False):
    if test:
        return mock_audio_stream_generator_with_real_audio()
    return generate(text=text, stream=True, api_key=api_key)


def mock_audio_stream_generator_with_real_audio(file_path="test.mp3", chunk_size=2048):
    """
    A generator to simulate streaming audio data from a real file in chunks.
    
    :param file_path: Path to the audio file to stream.
    :param chunk_size: Size of each chunk in bytes.
    """
    try:
        with open(file_path, 'rb') as audio_file:
            while True:
                chunk = audio_file.read(chunk_size)
                if not chunk:
                    break  # End of file
                yield chunk
    except FileNotFoundError:
        print(f"Audio file {file_path} not found.")


def mock_audio_stream_generator(text, chunk_size=2048, sleep_time=0.15):
    """
    A mock generator to simulate generating audio stream chunks.
    
    :param text: Text to generate mock audio from.
    :param chunk_size: Size of each chunk in bytes.
    :param sleep_time: Time to sleep between each yield statement in seconds.
    """
    words = text.split()
    total_size = len(words) * chunk_size
    num_chunks = len(words)
    print("MOCK AUDIO: ", end=" ", flush=True)
    for i in range(num_chunks):
        # Generate a chunk of bytes. In a real scenario, this would be actual audio data.
        # Here, we're just generating dummy data.
        chunk = b'\0' * chunk_size  # Placeholder for audio data chunk
        #print(f"Generated chunk {i + 1}/{num_chunks}")
        yield chunk
        print(words[i], end=" ", flush=True)
        time.sleep(sleep_time)
    # Handle any remaining bytes if total_size is not a multiple of chunk_size
    remaining_bytes = total_size % chunk_size
    if remaining_bytes:
        yield b'\0' * remaining_bytes
    print("")

if __name__ == "__main__":
    #audio_stream = generate_audio_stream("This is a... streaming voice!!")
    #print("test audio stream")
    #stream(audio_stream)

    # test the async version
    asyncio.run(__test_async())
    asyncio.run(__test_async_cancel())

