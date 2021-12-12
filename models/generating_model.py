import queue
import re
import heapq
import threading
import random
from torch.nn.functional import softmax
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer,
    BertForNextSentencePrediction)
from textblob import TextBlob, Word
import spacy


class GeneratingModel:
    def __init__(self):
        # default messages to send when stack of generated messages is empty
        self._init_default_messages = ["How are you?",
                                       "How is your day?",
                                       "Where have you been?",
                                       "Have you made any recent changes in your life?",
                                       "Do you have much free time during the day?",
                                       "What are you doing this weekend?",
                                       "What did you do last weekend?",
                                       "Where do you spend your free time?",
                                       "How do you like to spend your free time?",
                                       "Do you ever feel that you waste your free time?",
                                       "What makes you unique?"]
        self.default_messages = self._init_default_messages[:]
        random.shuffle(self.default_messages)
        self._generated_addons = ['What do you think about it?',
                                  'What does this means for you?']

        self._nlp = spacy.load("en_core_web_sm")

        # model and tokenizer for generating text
        self._gen_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self._gen_model = GPT2LMHeadModel.from_pretrained("gpt2",
                                                          pad_token_id=self._gen_tokenizer.eos_token_id)
        self._gen_length = 50  # amount of tokens to generate

        # model and tokenizer for next sentence prediction to filter generated messages
        self._nsp_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

        # queue of user's messages to process in another thread and put generated to stack
        self._message_queue = queue.Queue()
        self._generated_stack = queue.LifoQueue()
        self._processing_thread = threading.Thread(target=self._process, daemon=True,
                                                   name='model_processing_thread')

        # functions to filter generated text
        # new ones could be just added to this list
        # must return bool and accept 2 arguments:
        #  generated message and initial message
        self._filters = [self._phone_filter, self._nsp_filter]

        # pool of themes
        # it doesn't have to be locked
        self._themes_pool = []
        self._ask_theme_question = False
        self._theme_count_alfa = 3

        # priority to return this default questions from this queue
        self._default_messages_priority = queue.Queue()

    def _split_text(self, article):
        """Split article into sentences generator"""
        for sentence in self._nlp(article).sents:
            yield sentence.text

    def _get_aspects(self, sentences):
        for sentence in sentences:
            doc = self._nlp(sentence)
            descriptive_term = ''
            target = ''
            for token in doc:
                if token.dep_ == 'nsubj':
                    target = token.text
                if token.pos_ == 'ADJ':
                    prepend = ''
                    for child in token.children:
                        if child.pos_ != 'ADV':
                            continue
                        prepend += child.text + ' '
                    descriptive_term = prepend + token.text
            if target == '':
                target = doc.ents[0].text if doc.ents else 'something'
            aspect = {'aspect': target, 'description': descriptive_term,
                      'sentiment_polarity': TextBlob(sentence).sentiment.polarity}
            yield aspect, sentence

    def _get_nsp_probability(self, seq_a, seq_b):
        """Next sentence prediction"""
        inputs = self._nsp_tokenizer(seq_a, seq_b, return_tensors='pt')
        logits = self._nsp_model(**inputs).logits
        probability = softmax(logits, dim=1)[0][0].item()
        return probability

    def _generate_texts(self, aspects):
        """Text generation"""
        for aspect, message in aspects:
            sequence = aspect['aspect'] + ' ' + aspect['description'] + '. Indeed, '
            input_ids = self._gen_tokenizer.encode(sequence, return_tensors='pt')

            # using sampling
            output = self._gen_model.generate(input_ids,
                                              do_sample=True,
                                              max_length=self._gen_length,
                                              top_k=0,
                                              temperature=1.5)
            res = self._gen_tokenizer.decode(output[0], skip_special_tokens=True)
            yield res[len(sequence):], message

    def _phone_filter(self, gen, mes) -> bool:
        """Phone number filter"""
        p = re.compile('(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
        return not bool(p.match(gen))

    def _nsp_filter(self, gen, mes) -> bool:
        return self._get_nsp_probability(mes, gen) > 0.85

    def _check_themes(self, aspects):
        """Check if there is no constantly recurring theme"""
        for aspect, message in aspects:
            yield aspect, message

    def _filter_generated(self, generated):
        for g in generated:
            ret_f = True
            for f in self._filters:
                if not f(*g):
                    ret_f = False
                    break
            if ret_f:
                yield g

    def _enhance_generated(self, generated):
        for gen, message in generated:
            res = gen.strip()
            res = re.sub('\s+?', ' ', res)
            match = re.search('[!.?]', res[::-1])
            if match:
                res = res[: - match.end() + 1]
            else:
                res = res.rstrip(',:')
                res += '.'
                res = res.capitalize()
            res += '\n' + ''.join(random.sample(self._generated_addons, 1))
            yield res

    def start(self):
        """Start processing thread daemon.
        Should be called before using other methods"""
        if not self._processing_thread.is_alive():
            self._processing_thread.start()

    def _process(self):
        """Main processing loop"""
        while True:
            message = self._message_queue.get()
            sentences = self._split_text(message)
            aspects = self._get_aspects(sentences)
            theme_checked_aspects = self._check_themes(aspects)
            raw_generated = self._generate_texts(theme_checked_aspects)
            filtered_generated = self._filter_generated(raw_generated)
            enhanced_generated = self._enhance_generated(filtered_generated)
            for gen_message in enhanced_generated:
                self._generated_stack.put(gen_message)
            self._message_queue.task_done()

    def push(self, message):
        """Push user's messages to model"""
        self._message_queue.put(message)

    def get(self):
        """Retrieve messages from model"""
        # check firstly default messages priority
        try:
            result = self._default_messages_priority.get_nowait()
            self._default_messages_priority.task_done()
            return result
        except queue.Empty:
            # then check generated messages queue
            try:
                result = self._generated_stack.get_nowait()
                self._generated_stack.task_done()
                return result
            except queue.Empty:
                # else return default or specific question on one of least used themes
                if self._ask_theme_question and \
                        len(self._themes_pool) and \
                        self._themes_pool[0][0] < self._theme_count_alfa:
                    self._ask_theme_question = False
                    return f'What can you say about {self._themes_pool[0][1]}?'
                self._ask_theme_question = True
                return self.default_messages.pop()

    def set_default_messages_priority_for(self, amount):
        """Set priority for amount of default messages"""
        for i in range(amount):
            self._default_messages_priority.put(self.default_messages.pop())

    def get_parent_hyper(self, noun, level):
        if not noun or level < 0:
            return None
        word = Word(noun)
        if len(word.synsets) == 0:
            return None
        hp = None
        for syn in word.synsets:
            for lem_name in syn.lemma_names():
                if lem_name == noun:
                    if level >= len(syn.hypernym_paths()[0]):
                        return None
                    return syn.hypernym_paths()[0][-level].lemma_names()[0]
        return None

    def have_comm_base(self, w1, w2):
        max_level = 4
        a1 = []
        a2 = []
        for level in range(1, max_level):
            p1 = self.get_parent_hyper(w1, level)
            p2 = self.get_parent_hyper(w2, level)
            if p1:
                a1.append(p1.replace("_", " "))
            if p2:
                a2.append(p2.replace("_", " "))
        if any(i in a1 for i in a2):
            return a1[i]
        return False

    def comm_base(self, w1, w2):
        max_level = 4
        a1 = []
        a2 = []
        for level in range(1, max_level):
            p1 = self.get_parent_hyper(w1, level)
            p2 = self.get_parent_hyper(w2, level)
            if p1:
                a1.append(p1.replace("_", " "))
            if p2:
                a2.append(p2.replace("_", " "))
        comm_hyper = []
        for a in a1:
            for b in a2:
                if a == b:
                    comm_hyper.append(b)
        return comm_hyper
