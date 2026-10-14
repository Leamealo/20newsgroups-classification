import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

from src.data import load_dataset, TARGET_CATEGORIES
from src.models import build_ngram_pipeline
from src.metrics import evaluate_selective_classification

def main():
    print("1. Загружаем данные (3 класса: бейсбол, хоккей, космос)...")
    train_data, test_data = load_dataset()

    print("2. Обучаем модель (TF-IDF с униграммами и биграммами + Логистическая регрессия)...")
    model = build_ngram_pipeline()
    model.fit(train_data.data, train_data.target)

    preds = model.predict(test_data.data)
    base_f1 = f1_score(test_data.target, preds, average='macro')
    print(f"\n[Baseline] Macro-F1 на всех тестовых данных: {base_f1:.4f}")

    print("\n3. Проводим эксперимент с порогами уверенности (Selective Classification)...")
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]
    df_res = evaluate_selective_classification(model, test_data.data, test_data.target, thresholds)
    print("\nРезультаты по порогам уверенности:")
    print(df_res.to_string(index=False))

    print("\n4. Создаем матрицу ошибок (Confusion Matrix)...")
    cm = confusion_matrix(test_data.target, preds)

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_CATEGORIES, yticklabels=TARGET_CATEGORIES)
    plt.title('Confusion Matrix (Baseball vs Hockey vs Space)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    print("Матрица ошибок сохранена в папку results/confusion_matrix.png")

if __name__ == "__main__":
    main()
