import logging
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def sentiment_analyzer(text_to_analyze):
    """
    تحليل مشاعر النص المدخل باستخدام نموذج تحليل المشاعر المدرّب.

    الوسيطات:
        text_to_analyze (str): النص المدخل للتحليل.

    المخرجات:
        dict: قاموس يحتوي على تسمية المشاعر ('مشاعر إيجابية' أو 'مشاعر سلبية')
              ودرجة الثقة المرتبطة بها.
    """
    model_directory = "/Users/saad/Desktop/Workspace/Arabic_Sentiment_Analysis_flask/SentimentAnalysis/best_sa_model"

    try:
        pipe = pipeline(
            "sentiment-analysis",
            model=model_directory,
            # استخدام GPU إذا كان متاحًا، وإلا استخدام CPU
            device=0 if torch.cuda.is_available() else -1
        )
        result = pipe(text_to_analyze)[0]

        logging.debug(f"نتيجة الأنبوب: {result}")

        # التأكد من أن النتيجة تحتوي على المفاتيح المتوقعة
        if 'label' not in result or 'score' not in result:
            raise ValueError("مخرجات النموذج تفتقر إلى المفاتيح المتوقعة.")

        if result['label'] == "Positive":
            label = " مشاعر إيجابية "
        else:
            label = " مشاعر سلبية "

        # ترجمة درجة الثقة إلى العربية
        score_arabic = f" {result['score']} "

        return {
            ' التسمية ': label,
            ' درجة وثوق المشاعر في النص ': score_arabic
        }
    except Exception as e:
        logging.error(f"خطأ في تحليل المشاعر: {e}")
        return {
            'خطأ': str(e)
        }


# استخدام المثال
if __name__ == "__main__":
    text_to_analyze = "كيف حالك"
    response = sentiment_analyzer(text_to_analyze)
    print(response)