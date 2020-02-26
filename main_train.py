from fine_tuning_bert import main
from fine_tune_sbert import train_sbert



#############################################
corpus_input_file = 'data/corpus.txt'
bert_output_dir = 'output/bert'
model_type = 'bert'
model_name_or_path = 'bert-base-uncased'
sbert_output_dir = 'output/sbert'
#############################################

main(corpus_input_file, bert_output_dir, model_type, model_name_or_path)

train_sbert(bert_output_dir, sbert_output_dir)
