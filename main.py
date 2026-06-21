import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import re

# ==========================================
# 1. デバイスと設定
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

context_length = 10
embedding_dim = 64
hidden_dim = 128
batch_size = 512  # 大量データでもVRAMを溢れさせないためのバッチサイズ

# ==========================================
# 2. ファイルパスの指定（CUI仕様）
# ==========================================
# RunPod上のカレントディレクトリにある「training_data.txt」を読み込む想定
file_path = "training_data.txt" 

if not os.path.exists(file_path):
    print(f"❌ '{file_path}' が見つかりません。RunPodにファイルをアップロードしてください。")
    exit()

# ファイルの読み込みと前処理
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# クレンジング処理
clean_text = re.sub(r"《.*?》", "", raw_text)
clean_text = re.sub(r"［.*?］", "", clean_text)
clean_text = clean_text.replace("\n", "").replace("\r", "").replace(" ", "").replace("\u3000", "")

unwanted_chars = ["！", "？", "（", "）", "「", "」", "・", "。", "、"]
for char in unwanted_chars:
    clean_text = clean_text.replace(char, "")

print(f"元の文字数: {len(raw_text)}文字 -> クレンジング後: {len(clean_text)}文字")

# ==========================================
# 3. データの準備
# ==========================================
text = clean_text
chars = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
print(f"語彙数: {vocab_size}")

x_data = []
y_data = []
for i in range(len(text) - context_length):
    context = text[i:i+context_length]
    target = text[i+context_length]
    x_data.append([char_to_int[c] for c in context])
    y_data.append(char_to_int[target])

X = torch.tensor(x_data, dtype=torch.long)
Y = torch.tensor(y_data, dtype=torch.long)

# DataLoaderを使ってデータを小分け（ミニバッチ化）にする
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================================
# 4. モデル定義
# ==========================================
class MassLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, hidden_dim):
        super(MassLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_length * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.embedding(x)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = MassLLM(vocab_size, embedding_dim, context_length, hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ==========================================
# 5. 自動学習回数（Early Stopping / 早期終了）
# ==========================================
print("\n🚀 学習を開始します...")
max_epochs = 2000
patience = 50  
best_loss = float('inf')
patience_counter = 0

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0.0
    
    # バッチごとに学習を進める
    for batch_X, batch_Y in dataloader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)
    
    current_loss = epoch_loss / len(dataset)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {current_loss:.4f}")
    
    if current_loss < best_loss - 0.0005:  
        best_loss = current_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"🛑 Lossの減少が頭打ちになったため、{epoch+1} エポックで自動停止しました。(Best Loss: {best_loss:.4f})")
        break

if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth"))

# ==========================================
# 6. テキスト入力と生成（CUI仕様）
# ==========================================
print("\n--- AIによる自動生成 ---")
model.eval()

# ターミナルから直接文字を入力してもらう形に変更
prompt = input(f"AIに喋らせたい最初の文字を自由に文字入力してください（例: わがはいは）：")

if not prompt:
    print("入力がなかったため、デフォルトのテキストを使用します。")
    prompt = text[:context_length]

filtered_prompt = "".join([c for c in prompt if c in char_to_int])
if len(filtered_prompt) < context_length:
    filtered_prompt = (text[:context_length] + filtered_prompt)[-context_length:]
else:
    filtered_prompt = filtered_prompt[-context_length:]

current_context = filtered_prompt
result = prompt  
temperature = 1.0

with torch.no_grad():
    for _ in range(100):
        x_input = [char_to_int[c] for c in current_context]
        x = torch.tensor([x_input], dtype=torch.long).to(device)
        
        logits = model(x) / temperature
        probabilities = torch.softmax(logits, dim=1)
        
        next_char_id = torch.multinomial(probabilities, num_samples=1).item()
        next_char = int_to_char[next_char_id]
        
        result += next_char
        current_context = current_context[1:] + next_char

print(f"\n【生成されたテキスト】\n{result}")