from summarizer.summarize import generate_summary
from action_item_classifier.pred import pred_action_items

filename = input("Enter filename here (path/to/text/file.txt): ")
# input_text = ""
f = open(filename, 'r')
input_text = f.read()

summary = generate_summary(input_text)
action_items = pred_action_items(summary)
print('SUMMARY:', summary)
if len(action_items) == 0:
    print("No action items!")
else:
    print('ACTION ITEMS:', action_items)