from src.data import load_dataset
from src.models import build_ngram_pipeline
from src.metrics import evaluate_selective_classification
from sklearn.metrics import f1_score

def main():
    print("Загружаем данные...")
    train_data, test_data = load_dataset()

    print("Обучаем модель...")
    model = build_ngram_pipeline()
    model.fit(train_data.data, train_data.target)

    preds = model.predict(test_data.data)
    base_f1 = f1_score(test_data.target, preds, average='macro')
    print(f"Базовый Macro-F1 на всех данных: {base_f1:.4f}")

    print("\nПроводим эксперимент с порогами уверенности (Selective Classification):")
    thresholds = [0.50, 0.70, 0.85, 0.90, 0.95]
    df_res = evaluate_selective_classification(model, test_data.data, test_data.target, thresholds)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
