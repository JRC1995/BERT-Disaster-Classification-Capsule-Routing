Run using:

python train.py --model=[ENTER MODEL NAME HERE]


Example:
python train.py --model=BERT_BiLSTM


Available Models:

1) BERT_NL
2) BERT_BiLSTM
3) BERT_BiLSTM_attn
4) BERT_attn_BiLSTM
5) BERT_attn_BiLSTM_attn
6) BERT_capsule_BiLSTM_attn
7) BERT_capsule_BiLSTM_capsule
8) BERT_capsule


BERT_NL = BERT-MLP
BERT_BiLSTM = BERT-BiLSTM
BER_BiLSTM_attn = BERT-BiLSTM + attention-based-BiLSTM-hidden-state-aggregation
BERT_attn_BiLSTM = BERT-BiLSTM + attention-based-BERT-layer-aggregation
BERT_attn_BiLSTM_attn = BERT-BiLSTM + attention-based-BERT-layer-aggregation + attention-based-BiLSTM-hidden-state-aggregation
BERT_capsule_BiLSTM_attn = BERT-BiLSTM + capsule_routing_based-BERT-layer-aggregation + attention-based-BiLSTM-hidden-state-aggregation
BERT_capsule_BiLSTM_capsule = BERT-BiLSTM + capsule_routing_based-BERT-layer-aggregation + capsule_routing_based-BiLSTM-hidden-state-aggregation
BERT_capsule = BERT + capsule-routing-based-layer-aggregation + capsule-routing-based-BERT-layer-aggregated-hidden-states aggregation.


