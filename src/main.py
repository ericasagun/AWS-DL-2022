from summarizer.summarize import generate_summary
from action_item_classifier.pred import pred_action_items

filename = input("Enter filename here (path/to/text/file.txt): ")
f = open(filename, 'r')
input_text = f.read()

summary = generate_summary(input_text)
print()
action_items = pred_action_items(summary)
print('\nSUMMARY:', summary)
if not action_items:
    print("\nNo action items!")
else:
    print('\nACTION ITEMS:', action_items)
