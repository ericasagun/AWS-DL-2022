from summarizer.summarize import generate_summary
from action_item_classifier.pred import pred_action_items

filename = input("Enter filename here (path/to/text/file.txt): ")
f = open(filename, 'r')
input_text = f.read()

summary = generate_summary(input_text)
action_items = pred_action_items(summary)
print('SUMMARY:', summary)
if not action_items:
    print("No action items!")
else:
    print('ACTION ITEMS:', action_items)
