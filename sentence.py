import string
import logging

# See: https://github.com/bminixhofer/wtpsplit

from wtpsplit import WtP

wtp = WtP("wtp-canine-s-1l-no-adapters")
WTP_SENTENCE_THRESHOLD = 1.0e-5
SHORT_WORDS = ["yes", "no", "ok", "okay", "sure", "what", "yeah", "hi", "hello", "bye", "thanks", "cool", "gotit", "gotcha", "got it"]
THINKING_WORDS = ["hmm", "uh", "um", "huh", "hmm", "hmmm", "uhh", "uhm", "uhhh", "uhmmm", "ahem"]
NO_PUNCTUATION = str.maketrans('', '', string.punctuation)

logger = logging.getLogger("sentence")

def remove_punctuation(text):
    return text.translate(NO_PUNCTUATION)


def is_ignored_words(text):
    if text in THINKING_WORDS:
        return False


def simplify_text(text):
    return remove_punctuation(text.strip().lower())


def is_full_sentence(text):
    text = text.strip()
    simple_text = simplify_text(text)
    word_count = len(simple_text.split())

    # too short
    if len(text) <= 1:
        return False   
    # doesn't end in ellipse
    if text.endswith("..."):
        return False
    # thinking words only
    if is_ignored_words(simple_text):
        return False
    
    threshold = WTP_SENTENCE_THRESHOLD
    if simple_text in SHORT_WORDS:
        logging.debug(f"Short word detected: {simple_text}")
        return True
    # short sentence
    if simple_text not in SHORT_WORDS and (len(simple_text) < 10 or word_count < 3):
        threshold *= 10
    # ends in punctuation [disabled for now since assembly always adds punctuation]
    # if text[-1] in [".", "!", "?"]: 
    #     threshold *= 0.75
    #     if word_count > 4: # ends and is long
    #         threshold *= 0.5
    # check for full sentence using WtP
    # use the simple text version to ignore bad punctionation from transcriber
    prob = wtp.predict_proba(simple_text)
    # compare the probability that the final token is the end of sentence to the adjusted threshold
    logging.debug(f"Sentence prob: {prob[-1]} ( > {threshold}) : {prob[-1] > threshold}")
    #print(f"Sentence prob: {prob[-1]} ( > {threshold}) : {prob[-1] > threshold}")
    return prob[-1] > threshold


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("Is full sentence:")
    print("Hello", is_full_sentence("Hello"))
    print("Hello, how are you?", is_full_sentence("Hello, how are you?"))
    print("Hello, how are you", is_full_sentence("Hello, how are you"))
    print("Hello, how are", is_full_sentence("Hello, how are"))
    print("Hello, how are.", is_full_sentence("Hello, how are."))
          