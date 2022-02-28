from simplet5 import SimpleT5
import os 
os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Loading fine-tuned T5 model")
model = SimpleT5()
best_model_path = [dir for dir in os.listdir("models/") if os.path.isdir("models/" + dir)][-1]
model.load_model("t5" ,"models/" + best_model_path, use_gpu=False)
print("Model loaded")

def generate_summary(text):
    """
    Generate summary given a conversational text

    Args:
        text (str)

    Returns:
        summary (str)
    """
    summary = model.predict(text)[0]
    return summary
