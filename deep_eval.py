from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric
import pandas as pd
from deepeval.models import GeminiModel


EVAL_MODEL = "gemini-2.0-flash"
GOOGLEAI_API_KEY = "AIzaSyBcOvY7oDyy1L_BC4H3NwMyI9Woi060WdM"

eval_model = GeminiModel(model_name=EVAL_MODEL, api_key=GOOGLEAI_API_KEY)

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.8, model=eval_model,verbose_mode=True)
# contextual_precision = ContextualPrecisionMetric(threshold=0.8, model=eval_model)



dataset = EvaluationDataset()
dataset.add_test_cases_from_csv_file(
    file_path="/Users/ahmedmostafa/Downloads/eval_Korean_qa/assets/bench_korean.csv",
    input_col_name="input",
    actual_output_col_name="expected_output",
)
# Same as before, using the evaluate function
evaluate(dataset.test_cases, [answer_relevancy_metric])

# Or, use the evaluate method directly, they're exactly the same
# dataset.evaluate([answer_relevancy_metric])