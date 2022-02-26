from summarizer.summarize import generate_summary
from action_item_classifier.pred import pred_action_items

# input_text = input("Enter text here: ")
input_text = "Hal: Have you got any homework for tomorrow? Amy: no dad Hal: ru sure? Amy: told ya Hal: You know mum's not home today. Amy: I know, I can use the microwave Hal: good girl. I'll be home around 6 Amy: yeah right Hal: what do you mean Amy: sry dad but you're never ever home at 6 Hal: i suppose you;re right. but I'll try today Amy: ok. can I go to Alex? Hal: ok, but be back home before 7. we'll have dinner together Amy: ok dad Hal: and if you really have no homework to do Amy: sure thing dad"
summary = generate_summary(input_text)
action_items = pred_action_items(summary)
print(summary)
if len(action_items) == 0:
    print("No action items!")
else:
    print(action_items)