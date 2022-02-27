from datasets import load_dataset, load_metric, Dataset
import pandas as pd
from simplet5 import SimpleT5
from rouge import Rouge
import os
import random

rouge = Rouge()

def to_dataframe(dataset):
    """
    Process downloaded data into format required by T5

    Args:
        dataset (pd.DataFrame): downloaded dataset

    Returns:
        df (pd.DataFrame): processed dataset
    """
    columns = list(dataset.features.keys())
    df = pd.DataFrame()

    for column in columns:
        df[column] = dataset[column]
        df[column] = df[column].str.replace('\r', ' ')
        df[column] = df[column].str.replace('\n', ' ')

    # simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
    df = df.rename(columns={"summary":"target_text", "dialogue":"source_text"})

    # T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
    df['source_text'] = "summarize: " + df['source_text']

    return df

# samsum dataset - https://huggingface.co/datasets/samsum
print("Downloading data...")
train_dataset = load_dataset("samsum", split="train")
test_dataset = load_dataset("samsum", split="test")
valid_dataset = load_dataset("samsum", split="validation")
print("Data downloaded!")

print("Process data...")
train_df = to_dataframe(train_dataset)
test_df = to_dataframe(test_dataset)
valid_df = to_dataframe(valid_dataset)
print("Data processing done!")

# to save dataset, uncomment the following lines
# train_df.to_csv("datasets/train.csv")
# test_df.to_csv("datasets/test.csv")
# valid_df.to_csv("datasets/valid.csv")

print("Downloading T5 model...")
model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")  # model_name options: t5-small, t5-large, t5-3b, t5-11b
print("T5 model downloaded!")

# fine tune and save T5
print("Fine tuning T5 model...")
model.train(train_df=train_df,
            eval_df=test_df, 
            source_max_token_len=128,
            target_max_token_len=50,  
            batch_size=8, 
            max_epochs=3, 
            use_gpu=False, # set to false?
            outputdir="models/"
            )
print("Fine tuning T5 model done!")

best_model_path = [dir for dir in os.listdir("models/") if os.path.isdir(dir)][-1]
model.load_model("models/" + best_model_path, use_gpu=False)

print("Testing T5 model...")
convos = test_df["source_text"]
references = test_df["target_text"]
hypotheses = []
for convo in convos:
    hypotheses.append(model.predict(convo)[0])
print("Testing T5 model done!")

scores = rouge.get_scores(hypotheses, list(references), avg=True)
print("Fine-tuned model performance:")
print(scores)

print("Validating T5 model...")
random_convos = random.sample(list(valid_df["source_text"]), 3)  # sample 3 random convos
for convo in random_convos:
    print("Conversation:")
    print(convo)
    print("Generated summary:")
    print(model.predict(convo)[0])
    print("####################################################################")
