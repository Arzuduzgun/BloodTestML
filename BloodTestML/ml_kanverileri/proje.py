import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QComboBox, QPushButton, QLabel, QWidget
from PyQt5.QtCore import Qt

# Karışıklık matrisini görselleştiren fonksiyon
def plot_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    plt.title('Karışıklık Matrisi')
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()

# Veri seti işlemleri ve model eğitimi
def process_and_train(data, model_name, scale_data, balance_method=None):
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Dengesizlik ile mücadele (RUS veya ROS)
    if balance_method == "RUS":
        rus = RandomUnderSampler(random_state=42)
        X, y = rus.fit_resample(X, y)
    elif balance_method == "ROS":
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)

    # Veri normalizasyonu (scale_data parametresine göre)
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Eğitim ve test veri setini ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model seçimi
    if model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("Geçersiz model seçimi")

    # Modeli eğitme
    model.fit(X_train, y_train)

    # Tahminler yapma
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # Probabilistik tahminler

    # Performans metriklerini hesaplama
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1])  # Özgüllük
    f1 = f1_score(y_test, y_pred)

    return cm, accuracy, sensitivity, specificity, f1, y_proba

# K-fold çapraz doğrulama
def k_fold_cross_validation(data, model_name, k=3, scale_data=False):
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Veri normalizasyonu (scale_data parametresine göre)
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # KFold kullanımı
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    model = None

    if model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'KNN':
        model = KNeighborsClassifier()
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("Geçersiz model seçimi")

    # K-fold çapraz doğrulama skorları
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

    return scores

# Orijinal veri seti
original_data = pd.read_excel(r"C:\Users\arzud\OneDrive\Masaüstü\214410047_ArzuDuzgun\214410047_ArzuDuzgun_Proje\Kan_Verileri_Dataset.xlsx")

# Gürültülü veri seti
gurultulu_data = original_data.copy()
gurultulu_data.iloc[:, :-1] += np.random.normal(0, 0.1, original_data.iloc[:, :-1].shape)

# Gürültülü veri üzerinde forward fill ve backward fill işlemi
gurultulu_data_ffill = gurultulu_data.fillna(method='ffill')  # Forward fill
gurultulu_data_bfill = gurultulu_data_ffill.fillna(method='bfill')  # Backward fill

# Normalize edilmiş veri seti
scaler = StandardScaler()
normalized_data = original_data.copy()
normalized_data.iloc[:, :-1] = scaler.fit_transform(normalized_data.iloc[:, :-1])

# PyQt5 Arayüzü
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Model Seçimi ve Değerlendirmesi')
        self.setGeometry(100, 100, 600, 500)

        # Arayüz elemanlarını oluşturma
        self.layout = QVBoxLayout()

        self.dataset_combobox = QComboBox()
        self.dataset_combobox.addItems([
            "Orijinal", 
            "Dengesiz (RUS)", 
            "Dengesiz (ROS)", 
            "Gürültülü", 
            "Gürültülü (Fill Edilmiş)", 
            "Normalize Edilmiş"
        ])

        self.model_combobox = QComboBox()
        self.model_combobox.addItems(["Random Forest", "KNN", "Decision Tree"])

        self.result_label = QLabel("Sonuçlar burada görünecek")
        self.result_label.setAlignment(Qt.AlignTop)

        self.proba_label = QLabel("Olasılık Sonuçları burada görünecek")
        self.proba_label.setAlignment(Qt.AlignTop)

        self.layout.addWidget(QLabel("Veri Seti Seçin:"))
        self.layout.addWidget(self.dataset_combobox)
        self.layout.addWidget(QLabel("Model Seçin:"))
        self.layout.addWidget(self.model_combobox)
        self.result_label.setWordWrap(True)
        self.proba_label.setWordWrap(True)

        self.run_button = QPushButton("Sonuçları Hesapla")
        self.run_button.clicked.connect(self.run_model)

        self.kfold_button = QPushButton("K-Fold Sonuçları Hesapla (K=5)")
        self.kfold_button.clicked.connect(self.run_kfold)

        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.kfold_button)
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.proba_label)

        # Ana pencereyi oluşturma
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def get_selected_data(self):
        dataset_type = self.dataset_combobox.currentText()
        if dataset_type == "Orijinal":
            return original_data, None
        elif dataset_type == "Dengesiz (RUS)":
            return original_data, "RUS"
        elif dataset_type == "Dengesiz (ROS)":
            return original_data, "ROS"
        elif dataset_type == "Gürültülü":
            return gurultulu_data, None
        elif dataset_type == "Gürültülü (Fill Edilmiş)":
            return gurultulu_data_bfill, None  # İşlenmiş gürültülü veri
        elif dataset_type == "Normalize Edilmiş":
            return normalized_data, None
        else:
            raise ValueError("Geçersiz veri seti seçimi")

    def run_model(self):
        model_name = self.model_combobox.currentText()
        data, balance_method = self.get_selected_data()

        # Modeli çalıştırma
        cm, accuracy, sensitivity, specificity, f1, y_proba = process_and_train(
            data, model_name, scale_data=False, balance_method=balance_method
        )

        # Sonuçları güncelleme
        self.result_label.setText(f"Doğruluk: {accuracy:.2f}\nDuyarlılık: {sensitivity:.2f}\nÖzgüllük: {specificity:.2f}\nF1 Skoru: {f1:.2f}")

        # İlk 5 olasılık sonuçlarını gösterme
        proba_str = "İlk 5 Örnek Olasılıkları:\n"
        for i in range(min(5, len(y_proba))):  # İlk 5 örnek
            proba_str += f"Örnek {i+1}: {y_proba[i]}\n"
        self.proba_label.setText(proba_str)

        # Karışıklık matrisini görselleştirme
        plt.figure(figsize=(6, 6))
        plot_confusion_matrix(cm)
        plt.show()

    def run_kfold(self):
        model_name = self.model_combobox.currentText()
        data, _ = self.get_selected_data()

        # K-Fold işlemi
        k = 5  # K=5
        scores = k_fold_cross_validation(data, model_name, k)

        # K-Fold sonuçlarını ekrana yazdırma
        kfold_results = f"K-Fold Sonuçları (K={k}):\n"
        for i, score in enumerate(scores):
            kfold_results += f"Fold {i + 1}: {score:.2f}\n"

        average_score = sum(scores) / len(scores)
        kfold_results += f"\nOrtalama Skor: {average_score:.2f}\n"
        # Sonuçları güncelleme
        self.result_label.setText(f"{kfold_results}")

# Ana uygulama başlatma
def main():
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()