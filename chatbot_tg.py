from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import torch.nn as nn
from helpers import indexesFromSentence, normalizeString, loadPrepareData
from datetime import datetime
from modules import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder
import telebot
from telebot import types


token = '5458490557:AAEnX6NO1NnWtokgOeKt72tDyvg1Sdej8GM'

bot = telebot.TeleBot(token)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

PAD_token = 0  
SOS_token = 1 
EOS_token = 2

MAX_LENGTH = 32

global output_words

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    tokens, scores = searcher(input_batch, lengths, MAX_LENGTH)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc, input_sentence):
    while(1):
        try:
            
            if input_sentence.lower() == '/stop':
                return('Пока!')
                break

            input_sentence = normalizeString(input_sentence)
            
            found = evaluate(encoder, decoder, searcher, voc, input_sentence)
            
            found[:] = [x for x in found if not (x == 'EOS' or x == 'PAD')]

            output_words = ' '.join(found)

            return(output_words)
            
            break

        except KeyError:
            
            output_words = "Я не понял некоторые слова :("
            
            return(output_words)

            break



def main():

    corpus_name = "qa"
    #datafile = 'drive/MyDrive/Chatbot-with-Pytorch/train.txt'
    datafile = input('Введите путь до файла train.txt:')

    save_dir = 'save'
    voc, pairs = loadPrepareData(corpus_name, datafile, save_dir)

    model_name = 'cb_model'
    dropout = 0

    attn_model = 'dot'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0
    batch_size = 64

    loadFilename = None
    checkpoint_iter = 8000

    loadFilename = os.path.join(save_dir, model_name, corpus_name,
                                '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                                '{}_checkpoint.tar'.format(checkpoint_iter))

    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    embedding = nn.Embedding(voc.num_words, hidden_size)

    if loadFilename:
        embedding.load_state_dict(embedding_sd)    
    
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers,
                                  dropout)
    
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('The bot is ready to work')

    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder)
    
    @bot.message_handler(commands=['start','help'])
    def start_message(message):
	    bot.send_message(message.chat.id,'Привет!\nЯ глупый бот, который не очень хорошо отвечает на вопросы, но я очень стараюсь.\nПопробуй написать что-нибудь!')

    @bot.message_handler(content_types=['text'])
    def get_message(message):
        output_words = evaluateInput(encoder, decoder, searcher, voc, message.text)
        bot.send_message(message.chat.id, ''.join(output_words))

    bot.polling(none_stop=True, interval=0)


if __name__ == '__main__':
    main()