import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

print("--- 1. VERIYI TANIMA VE YUKLEME ---")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0: Risk Var (Evet), 1: Risk Yok (Hayır)

print(f"Toplam Satır ve Sütun Sayısı: {df.shape}")
print("Sınıf Dağılımı (Kaç Evet / Kaç Hayır):")
print(df['target'].value_counts())

print("\n--- 2. VERI TEMIZLEME ---")
print("Eksik değer kontrolü yapıldı, temiz veri seti kullanılıyor.")
df = df.dropna()

print("\n--- 3. GORSELURSTURME ---")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df['target'].value_counts().plot(kind='bar', ax=axes[0], color=['salmon', 'skyblue'])
axes[0].set_title('Hedef Sınıf Dağılımı')
axes[0].set_xlabel('Sınıflar (0: Risk, 1: Temiz)')
axes[0].set_ylabel('Kişi Sayısı')

df.iloc[:, 0].plot(kind='hist', ax=axes[1], bins=20, color='purple')
axes[1].set_title('Özellik Dağılımı (Mean Radius)')
plt.tight_layout()
plt.savefig('grafikler.png')
print("Grafikler başarıyla 'grafikler.png' olarak kaydedildi.")

print("\n--- 4. MODEL EGITIMI VE TESTI ---")
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n--- 5. SONUÇLAR VE RAPORLAMA ---")
print(f"Accuracy (Doğruluk): {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (Kesinlik): {precision_score(y_test, y_pred):.4f}")
print(f"Recall (Duyarlılık): {recall_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n[BAŞARILI] Proje tamamlandı!")