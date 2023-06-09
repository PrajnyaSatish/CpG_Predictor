from flask import Flask, request, jsonify, render_template
from CpG_Predictor import CpGPredictor
import torch
from utils import dnaseq_to_intseq

app = Flask(__name__)



@app.route('/')
def index():
    return render_template("index.html")

# some config
LSTM_HIDDEN = 64
LSTM_LAYER = 6
EMBED_DIM=10
batch_size = 384
learning_rate = 0.5
dropout_p=0.4
epoch_num = 100
vocab_size = len("NACGT")

predict_model = CpGPredictor(LSTM_LAYER, LSTM_HIDDEN, dropout_p, 6, EMBED_DIM)
predict_model.load_state_dict(torch.load("models/padded_seq_model_weights.pth"))
predict_model.eval()

@app.route('/process_dna_sequence', methods=['POST'])
def process_dna_sequence():
    dna_seq = request.form.get('dna_seq')

    # Logic to count CGs in the DNA sequence
    original_count = dna_seq.count('CG')
    int_seq = list(dnaseq_to_intseq([el for el in dna_seq]))
    tester = torch.tensor(int_seq).unsqueeze(0)
    prediction = predict_model(tester)
    predicted_count = torch.round(prediction).item()
    print(predicted_count)
    response = {
        'count1': original_count,
        'count2': predicted_count
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
