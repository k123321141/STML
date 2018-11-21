
import argparse
import re
import sys
import pickle
import torch
reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-m','--model', type=str, default='rnn', choices=['rnn','attention'])
    parser.add_argument('-l','--language', type=str, \
                        choices=['eng','ch'], default='eng', \
                        help='Specify the language eng/ch to parse. eng by default.')
    args = parser.parse_args()
    
#    
    with open('tokenizer','rb') as f:
        tokenizer = pickle.load(f)
    with open(args.input,'rb') as f:
        ls = f.readlines()
    for l in ls:
        l = l.decode('utf8')
    if args.model == 'rnn':
        from src.Seq2seq import Seq2seq
        model = torch.load('./src/best.pt')
        
    

    
#     pat = re.compile("([a-zA-Z]+-*[a-zA-Z]+)") if args.language == 'eng' else re.compile(u"([\u4e00-\u9fff]+)")
#     with open(args.output, 'wb',) as f:
#         for s in ls:
#             if type(s) != str:
#                 s = str(s)
#             s = s.decode('utf8')
#             r = u''
#             for idx, text in enumerate(pat.findall(s)):
#                 r += text + u' '
#             f.write(r.strip()+u'\n')
