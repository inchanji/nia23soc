
epochs = 150
lr = 0.00006
batch_size = 16

label2id = {
    "crack":1,
    "reticular crack":2,
    "detachment":3,
    "spalling":4,
    "efflorescence":5,
    "leak":6,
    "rebar":7,
    "material separation":8,
    "exhilaration":9,
    "damage":10
}

id2label = {v: k for k, v in label2id.items()}
