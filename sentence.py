import re
import string
import logging

# See: https://github.com/bminixhofer/wtpsplit

from wtpsplit import WtP

wtp = WtP("wtp-canine-s-1l-no-adapters")
WTP_SENTENCE_THRESHOLD = 1.0e-5
SHORT_WORDS = ["yes", "yep", "yup", "ok", "okay", "sure", "yeah", "oh yeah", 
               "uhuh", "uh-uh", "uh uh", "uh-huh", "uh huh", "uhhuh",
               "no", "nope", "nah", "not really", "not sure", "not really sure",
               "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
               "bye", "goodbye", "good bye", "goodnight", "good night",
               "thanks", "cool", "gotit", "gotcha", "got it", 
               "great", "awesome", "alright", "alrighty", "alrighty then", 
               "nice", "fine", "fine by me", "fine with me",
               "what", "how", "how's that"
               ]
QUESTION_WORDS = ["what", "how", "why", "where", "when", "who", "which", "is"]
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
    # remove thinking words, but only full matches
    for word in THINKING_WORDS:
        simple_text = re.sub(r"\b" + word + r"\b", "", simple_text)
    simple_text = simple_text.strip()

    threshold = WTP_SENTENCE_THRESHOLD
    if simple_text in SHORT_WORDS:
        logging.debug(f"Short word detected: {simple_text}")
        return True
    # first sentence is short
    first_sentence = simplify_text(re.split("[.!?]+", text)[0])
    if first_sentence in SHORT_WORDS:    
        logging.debug(f"Short word detected in first sentence: {simple_text}")
        return True
    # short sentence
    if simple_text not in SHORT_WORDS and (len(simple_text) < 10 or word_count < 3):
        threshold *= 10
    # ends in punctuation [disabled for now since assembly always adds punctuation]
    #if text[-1] in [".", "!", "?"]: 
    if text[-1] == "?": # allow for questions for better conversation
        threshold *= 0.5
        if word_count > 4: # ends and is long
            threshold *= 0.2
        # starts with question word or short word
        if simple_text.split()[0] in QUESTION_WORDS + SHORT_WORDS:
            threshold *= 0.1
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
    print("So what is this thing?", is_full_sentence("So what is this thing?"))
    print("What do you think of this?", is_full_sentence("What do you think of this?"))
    print("Hey! What's up?", is_full_sentence("Hey! What's up?"))
    print("Uhm, okay.", is_full_sentence("Uhm, okay."))
          